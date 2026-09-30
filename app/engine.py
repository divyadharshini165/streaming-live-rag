"""Event-driven Streaming Live RAG engine.

on_chunk(session, text, t)   -> controller decision for each incoming transcript chunk
end_utterance(session, t)    -> structured output record (answer, citations, uncertainty, ...)
run_utterance(session, text) -> simulate a spoken utterance chunk-by-chunk (transcript replay)
"""
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor

from .config import Config
from .controller import RETRIEVE, SUPPRESS, WAIT, RetrievalController
from .corpus import load_corpus
from .decomposer import Decomposer
from .llm import LLM
from .retriever import Hit, HybridRetriever, rrf_merge
from .session import AnswerVersion, SessionStore
from .simulator import chunk_transcript
from .synthesizer import SUPPORT_THRESHOLD, Synthesizer
from .telemetry import Telemetry
from .text import content_terms, jaccard

log = logging.getLogger(__name__)
COVER_T = 0.7


class TurnState:
    def __init__(self, turn_id):
        self.turn_id = turn_id
        self.text = ""
        self.results = {}          # query -> list[Hit]
        self.retrieval_events = []
        self.suppressed = False
        self.first_retrieval_t = None
        self.timeline = []
        self.prefer_docs = frozenset()


class StreamingRAGEngine:
    def __init__(self, cfg: Config | None = None, retriever: HybridRetriever | None = None):
        self.cfg = cfg or Config()
        t0 = time.perf_counter()
        if retriever is None:
            chunks = load_corpus(self.cfg.corpus_dir)
            retriever = HybridRetriever(chunks, self.cfg.retrieval_mode, self.cfg.embed_model)
        self.retriever = retriever
        self.index_ms = round((time.perf_counter() - t0) * 1000, 1)
        self.llm = LLM(self.cfg)
        self.controller = RetrievalController(self.retriever)
        self.decomposer = Decomposer(self.retriever, self.llm, self.cfg.decomposer_mode)
        self.synth = Synthesizer(self.retriever, self.llm)
        self.store = SessionStore()
        self.tel: dict[str, Telemetry] = {}
        self.turns: dict[str, TurnState | None] = {}
        self.pool = ThreadPoolExecutor(max_workers=8)

    # ------------------------------------------------------------------ sessions
    def new_session(self) -> str:
        s = self.store.new()
        self.tel[s.session_id] = Telemetry(self.cfg.log_dir, s.session_id)
        self.turns[s.session_id] = None
        self.tel[s.session_id].emit("session_start", corpus_chunks=len(self.retriever.chunks),
                                    retrieval_mode=self.retriever.mode, llm=self.llm.available,
                                    config=self._cfg_summary())
        return s.session_id

    def end_session(self, sid):
        self.store.drop(sid)
        self.turns.pop(sid, None)
        self.tel.pop(sid, None)

    def _cfg_summary(self):
        c = self.cfg
        return dict(streaming=c.streaming, decomposition=c.decomposition, refinement=c.refinement,
                    suppression=c.suppression, decomposer_mode=c.decomposer_mode, model=c.llm_model)

    def _turn(self, sid) -> TurnState:
        st = self.turns.get(sid)
        if st is None:
            s = self.store.get(sid)
            s.turn_count += 1
            st = TurnState(s.turn_count)
            self.turns[sid] = st
            self.tel[sid].start_turn(st.turn_id)
            self.tel[sid].emit("turn_start", has_prior_answer=s.has_answer)
        return st

    # ------------------------------------------------------------------ retrieval
    def _covered(self, q, st, thr=COVER_T):
        qt = content_terms(q)
        return any(jaccard(qt, content_terms(k)) >= thr for k in st.results)

    def _hits_for(self, q, st):
        if q in st.results:
            return st.results[q]
        qt = content_terms(q)
        best = max(st.results, key=lambda k: jaccard(qt, content_terms(k)), default=None)
        return st.results.get(best, [])

    def _dispatch(self, sid, st, queries, trigger, t):
        tel = self.tel[sid]
        queries = [q for q in dict.fromkeys(queries) if q and not self._covered(q, st)]
        if not queries:
            return []
        for q in queries:
            tel.emit("retrieval_started", t, query=q, trigger=trigger)

        def run(q):
            t0 = time.perf_counter()
            hits = self.retriever.search(q, self.cfg.top_k, st.prefer_docs)
            return q, hits, round((time.perf_counter() - t0) * 1000, 2)

        for q, hits, ms in self.pool.map(run, queries):  # sub-queries run in parallel
            st.results[q] = hits
            ev = {"timestamp_s": t, "query": q, "trigger": trigger, "latency_ms": ms,
                  "top_chunks": [h.chunk.chunk_id for h in hits[:3]]}
            st.retrieval_events.append(ev)
            tel.emit("retrieval_completed", t, **ev, hits=[h.brief() for h in hits])
        if st.first_retrieval_t is None:
            st.first_retrieval_t = t
        return queries

    # ------------------------------------------------------------------ streaming input
    def on_chunk(self, sid: str, text: str, t: float) -> dict:
        s, tel, st = self.store.get(sid), self.tel[sid], self._turn(sid)
        st.text = f"{st.text} {text}".strip()
        tel.emit("chunk_received", t, text=text)
        out = {"t": t, "chunk": text, "decision": WAIT, "reason": "", "dispatched": []}
        c = self.cfg
        if not c.streaming:
            out["reason"] = "baseline_waits_for_utterance_end"
        elif st.suppressed:
            out.update(decision=SUPPRESS, reason="presentation_restructure")
        else:
            dec, reason, q = self.controller.on_chunk(st.text, s.has_answer, bool(st.results))
            if dec == SUPPRESS and not c.suppression:
                dec, reason = WAIT, "suppression_disabled"
            out.update(decision=dec, reason=reason)
            kind = self.controller.classify(st.text, s.has_answer)
            refine_turn = kind == "late_detail" and c.refinement
            if refine_turn and s.current:
                st.prefer_docs = frozenset(cid.split(" ")[0] for cid in s.current.evidence_ids)
            if dec == SUPPRESS:
                st.suppressed = True
            elif dec == RETRIEVE:
                if refine_turn:
                    qs = self.decomposer.delta_queries(st.text, s.topic_queries)
                    out["dispatched"] = self._dispatch(sid, st, qs, "late_detail", t)
                else:
                    q = q or st.text
                    out["dispatched"] = self._dispatch(sid, st, [q], "provisional", t)
            # after the first retrieval, launch newly completed clauses in parallel
            if st.results and not st.suppressed and dec != RETRIEVE:
                if refine_turn:
                    qs = self.decomposer.delta_queries(st.text, s.topic_queries, complete_only=True)
                    trig = "late_detail"
                elif kind == "new_query" and c.decomposition:
                    qs = self.decomposer.rule_decompose(st.text, complete_only=True)
                    qs = qs if len(qs) >= 2 else []
                    trig = "multi_intent"
                else:
                    qs, trig = [], ""
                new = self._dispatch(sid, st, qs, trig, t) if qs else []
                if new:
                    out.update(decision=RETRIEVE, reason=f"{trig}_detected", dispatched=new)
        tel.emit("controller_decision", t, decision=out["decision"], reason=out["reason"],
                 dispatched=out["dispatched"])
        st.timeline.append(out)
        return out

    # ------------------------------------------------------------------ utterance end
    def end_utterance(self, sid: str, t: float, on_token=None) -> dict:
        s, tel, st = self.store.get(sid), self.tel[sid], self._turn(sid)
        c = self.cfg
        tel.emit("utterance_end", t, text=st.text)
        t_end = time.perf_counter()
        first = {}

        def token_cb(tok):
            if "ms" not in first:
                first["ms"] = round((time.perf_counter() - t_end) * 1000, 2)
                tel.emit("first_token", t, ttft_ms=first["ms"])
            if on_token:
                on_token(tok)

        if not c.streaming and not (c.suppression or c.refinement):
            kind = "new_query"  # baseline: classic RAG always retrieves
        else:
            kind = self.controller.classify(st.text, s.has_answer)
            if kind == "presentation" and not c.suppression:
                kind = "new_query"
            if kind == "late_detail" and not c.refinement:
                kind = "new_query"
        retrieval_required = kind in ("new_query", "late_detail")
        tel.emit("turn_classified", t, turn_type=kind, retrieval_required=retrieval_required)

        prev = s.current
        sub_queries, extra_usage, evidence = [], {"tokens_in": 0, "tokens_out": 0}, []
        new_version = None

        if kind == "presentation":
            prev_ev = {cid: s.evidence[cid] for cid in prev.citations if cid in s.evidence}
            res = self.synth.restructure(prev, st.text, prev_ev, token_cb)
            reason = "presentation_restructure"
        elif kind == "conversational":
            thanks = re.search(r"\b(thanks|thank you|got it|okay|ok|great|perfect|cool)\b", st.text, re.I)
            res = {"answer": "You're welcome. Anything else you'd like to check?" if thanks
                   else "I'm listening - what would you like to find in the documents?",
                   "citations": [], "uncertainty": None, "rejected_citations": [],
                   "grounding": self.synth.grounding("", {}), "tokens_in": 0, "tokens_out": 0}
            token_cb(res["answer"])
            reason = "no_retrievable_intent"
        elif kind == "late_detail":
            reason = "late_detail_refinement"
            st.prefer_docs = frozenset(cid.split(" ")[0] for cid in prev.evidence_ids)
            sub_queries = self.decomposer.delta_queries(st.text, s.topic_queries)
            self._dispatch(sid, st, sub_queries, "late_detail", t)
            plan = []
            for q in sub_queries:
                h = self._hits_for(q, st)
                plan.append((q, h, self.synth.support(q, h) >= SUPPORT_THRESHOLD))
            delta_ev = rrf_merge([p[1] for p in plan], cap=4)
            prior = [Hit(s.evidence[cid], 1.0, 1.0) for cid in prev.evidence_ids if cid in s.evidence]
            seen = {h.chunk.chunk_id for h in prior}
            evidence = prior + [h for h in delta_ev if h.chunk.chunk_id not in seen]
            tel.emit("fusion", t, prior_evidence=len(prior), delta_evidence=len(delta_ev),
                     evidence=[h.chunk.chunk_id for h in evidence])
            res = self.synth.refine(prev, st.text, plan, evidence, token_cb)
            s.topic_queries = s.topic_queries + sub_queries
        else:
            reason = "retrieval_required"
            if c.decomposition:
                sub_queries = self.decomposer.decompose(st.text)
                if self.decomposer.last_usage:
                    extra_usage = {k: self.decomposer.last_usage[k] for k in ("tokens_in", "tokens_out")}
            else:
                sub_queries = [st.text]
            tel.emit("decomposition", t, mode=c.decomposer_mode if c.decomposition else "off",
                     sub_queries=sub_queries)
            trig = "final" if not c.streaming else ("multi_intent" if len(sub_queries) > 1 else "final")
            self._dispatch(sid, st, sub_queries, trig, t)
            plan = []
            for q in sub_queries:
                h = self._hits_for(q, st)
                plan.append((q, h, self.synth.support(q, h) >= SUPPORT_THRESHOLD))
            tel.emit("sufficiency_check", t, checks=[{"sub_query": q, "supported": ok} for q, _, ok in plan])
            evidence = rrf_merge([p[1] for p in plan], cap=c.evidence_cap)
            tel.emit("fusion", t, candidates=sum(len(p[1]) for p in plan), evidence=[h.chunk.chunk_id for h in evidence])
            res = self.synth.answer(st.text, plan, evidence, token_cb)
            s.topic_queries = list(sub_queries)

        # ---- versioning & session memory
        lineage = None
        if kind in ("new_query", "late_detail"):
            for h in evidence:
                s.evidence[h.chunk.chunk_id] = h.chunk
            new_version = (prev.version + 1) if prev else 1
            s.versions.append(AnswerVersion(new_version, res["answer"], res["citations"], res["uncertainty"],
                                            sub_queries, [h.chunk.chunk_id for h in evidence], st.turn_id, kind))
            if prev and kind == "late_detail":
                pc, nc = set(prev.citations), set(res["citations"])
                lineage = {"kept": sorted(pc & nc), "added": sorted(nc - pc), "dropped": sorted(pc - nc)}
            tel.emit("answer_version", t, version=new_version, parent=prev.version if prev else None,
                     turn_type=kind, lineage=lineage)

        tel.emit("citation_check", t, citations=res["citations"], rejected=res.get("rejected_citations", []),
                 support_rate=res["grounding"]["support_rate"], sentences=res["grounding"]["sentences"])

        tin = res.get("tokens_in", 0) + extra_usage["tokens_in"]
        tout = res.get("tokens_out", 0) + extra_usage["tokens_out"]
        total_ms = round((time.perf_counter() - t_end) * 1000, 2)
        metrics = {
            "utterance_end_s": t,
            "first_retrieval_s": st.first_retrieval_t,
            "early_retrieval": st.first_retrieval_t is not None and st.first_retrieval_t < t,
            "lead_time_s": round(t - st.first_retrieval_t, 2) if st.first_retrieval_t is not None else None,
            "ttft_ms": first.get("ms", total_ms),
            "total_ms_after_end": total_ms,
            "retrieval_calls": len(st.retrieval_events),
            "tokens_in": tin, "tokens_out": tout,
            "cost_usd": round(self.llm.cost(tin, tout), 6),
        }
        tel.emit("turn_end", t, **metrics)

        record = {
            "session_id": sid, "turn_id": st.turn_id, "utterance": st.text,
            "turn_type": kind, "retrieval_required": retrieval_required, "reason": reason,
            "retrieval_events": st.retrieval_events, "sub_queries": sub_queries,
            "answer": res["answer"], "citations": res["citations"], "uncertainty": res["uncertainty"],
            "answer_version": new_version if new_version else (prev.version if prev else None),
            "parent_version": prev.version if (prev and new_version) else None,
            "evidence_ids": [h.chunk.chunk_id for h in evidence],
            "citation_lineage": lineage, "grounding": {k: v for k, v in res["grounding"].items() if k != "details"},
            "grounding_details": res["grounding"]["details"], "rejected_citations": res.get("rejected_citations", []),
            "metrics": metrics, "timeline": st.timeline,
        }
        self.turns[sid] = None
        return record

    # ------------------------------------------------------------------ transcript replay
    def run_utterance(self, sid: str, text: str, realtime: bool = False, on_chunk=None, on_token=None) -> dict:
        chunks, end_t = chunk_transcript(text, self.cfg.words_per_chunk, self.cfg.words_per_second,
                                         self.cfg.endpoint_silence_s)
        prev = 0.0
        for t, part in chunks:
            if realtime:
                time.sleep(max(0.0, t - prev))
                prev = t
            out = self.on_chunk(sid, part, t)
            if on_chunk:
                on_chunk(out)
        if realtime:
            time.sleep(max(0.0, end_t - prev))
        return self.end_utterance(sid, end_t, on_token)
