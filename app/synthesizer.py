"""Session-Aware Synthesis.

- Evidence sufficiency check per sub-query (IDF-weighted coverage) -> explicit uncertainty.
- LLM mode: streamed answer with inline [Doc_x §y] tags; invalid tags are stripped.
- Offline mode: extractive answer built from corpus sentences (grounded by construction).
- Grounding check: every sentence must cite a retrieved chunk that actually contains its terms.
"""
import math
import re
import time

from .text import attach_tags, content_terms, extract_citations, split_sentences

SUPPORT_THRESHOLD = 0.5
NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5}

SYS_ANSWER = (
    "You answer questions using ONLY the evidence passages provided. Rules:\n"
    "1. After every factual sentence write the tag(s) of the passage(s) it comes from, exactly as given, e.g. [Doc_07 §4].\n"
    "2. Never invent tags, numbers or facts that are not in the evidence.\n"
    "3. Address every sub-question in one short, unified answer (max 120 words, plain text, no headings).\n"
    "4. If the evidence does not answer a sub-question (always the case for ones marked NOT FOUND), do not guess; "
    "end with one line starting with 'UNCERTAIN:' that says what could not be verified from the corpus.")
SYS_REFINE = (
    "You update a previous answer after the user added a new detail. Use ONLY the evidence passages.\n"
    "1. Keep every statement from the previous answer that still holds, with its original tags.\n"
    "2. Change only what the new detail affects and add the new facts with their tags, e.g. [Doc_03 §3].\n"
    "3. Never invent tags or facts. Max 150 words, plain text.\n"
    "4. If the new detail cannot be verified from the evidence, end with a line starting with 'UNCERTAIN:'.")
SYS_FORMAT = (
    "Rewrite the previous answer exactly as the user asks (format, length, simpler wording). "
    "Do NOT add any fact that is not already in it. Keep the citation tags attached to the facts they support.")


class Synthesizer:
    def __init__(self, retriever, llm):
        self.r = retriever
        self.llm = llm
        n = len(retriever.chunks)
        self.unknown_idf = math.log((n + 1) / 0.5)

    # ---------- evidence sufficiency ----------
    def _w(self, term):
        return self.r.idf.get(term, self.unknown_idf)

    def support(self, query, hits):
        """IDF-weighted share of the query's terms found in its best evidence chunk."""
        q = [t for t in dict.fromkeys(content_terms(query)) if not t.isdigit()]
        if not q or not hits:
            return 0.0
        total = sum(self._w(t) for t in q)
        best = 0.0
        for h in hits[:3]:
            ct = set(content_terms(h.chunk.index_text))
            best = max(best, sum(self._w(t) for t in q if t in ct) / total)
        return best

    # ---------- grounding ----------
    def grounding(self, answer, evidence_map):
        results, invalid = [], []
        for s in split_sentences(answer.replace("\n", " ")):
            if s.upper().startswith("UNCERTAIN"):
                continue
            body = re.sub(r"\[[^\]]*\]", "", s).strip(" •-")
            terms = content_terms(body)
            if len(terms) < 2:
                continue  # connective / filler
            cites = extract_citations(s)
            bad = [c for c in cites if c not in evidence_map]
            invalid += bad
            best = 0.0
            for c in cites:
                if c in evidence_map:
                    ct = set(content_terms(evidence_map[c].index_text))
                    best = max(best, sum(1 for t in terms if t in ct) / len(terms))
            results.append({"sentence": body[:160], "citations": cites, "support": round(best, 2),
                            "supported": best >= 0.5})
        n = len(results)
        return {"sentences": n, "supported": sum(r["supported"] for r in results),
                "support_rate": round(sum(r["supported"] for r in results) / n, 3) if n else 1.0,
                "invalid_citations": invalid, "details": results}

    @staticmethod
    def _strip_invalid(text, valid):
        def fix(m):
            keep = [c for c in extract_citations(m.group(0)) if c in valid]
            return "[" + ", ".join(keep) + "]" if keep else ""
        return re.sub(r"\[[^\]]*Doc_[^\]]*\]", fix, text)

    @staticmethod
    def _split_uncertain(text):
        lines = text.strip().splitlines()
        unc = [l.split(":", 1)[1].strip() for l in lines if l.strip().upper().startswith("UNCERTAIN")]
        body = "\n".join(l for l in lines if not l.strip().upper().startswith("UNCERTAIN")).strip()
        return body, (" ".join(unc) or None)

    @staticmethod
    def _evidence_block(evidence):
        return "\n".join(f"[{h.chunk.chunk_id}] ({h.chunk.doc_title} - {h.chunk.section_title}): {h.chunk.text}"
                         for h in evidence)

    # ---------- extractive (offline) ----------
    def _best_sentences(self, query, hits, n=3, n_hits=2, skip=frozenset()):
        """Sentences that answer the query. Terms that only name the document (e.g. the venue name
        in its own fact sheet) count less, so the answer sentence beats the overview sentence."""
        q = set(content_terms(query))
        cands = []
        for rank, h in enumerate(hits[:n_hits]):
            title = set(content_terms(h.chunk.doc_title))
            section = set(content_terms(h.chunk.section_title))
            for s in split_sentences(h.chunk.text):
                if s in skip:
                    continue  # already used for another sub-query
                st = set(content_terms(s))
                score = sum(self._w(t) * (0.3 if t in title else 1.0) for t in q if t in st)
                score += 0.5 * sum(self._w(t) for t in q if t in section and t not in st)
                score *= 0.6 ** rank
                if score > 0:
                    cands.append((score, s, h.chunk.chunk_id))
        if not cands:
            return []
        cands.sort(key=lambda x: -x[0])
        best = cands[0][0]
        return [(s, cid) for sc, s, cid in cands[:n] if sc >= 0.6 * best]

    def _extractive(self, plan, n_hits=2, n=3):
        seen, parts = set(), []
        for sq, hits, ok in plan:
            if not ok:
                continue
            for s, cid in self._best_sentences(sq, hits, n=n, n_hits=n_hits, skip=frozenset(seen)):
                if s not in seen:
                    seen.add(s)
                    parts.append(f"{s.rstrip('.')} [{cid}].")
        return " ".join(parts)

    # ---------- public API ----------
    def answer(self, utterance, plan, evidence, on_token=None):
        """plan: list of (sub_query, hits, supported)."""
        t0 = time.perf_counter()
        emap = {h.chunk.chunk_id: h.chunk for h in evidence}
        missing = [sq for sq, _, ok in plan if not ok]
        usage = dict(tokens_in=0, tokens_out=0)
        if self.llm.available:
            sq_lines = "\n".join(f"{i+1}. {sq}{'' if ok else '  -> NOT FOUND in corpus'}"
                                 for i, (sq, _, ok) in enumerate(plan))
            msgs = [{"role": "system", "content": SYS_ANSWER},
                    {"role": "user", "content": f"User request: {utterance}\nSub-questions:\n{sq_lines}\n\n"
                                                f"Evidence:\n{self._evidence_block(evidence)}"}]
            res = self.llm.complete(msgs, max_tokens=350, on_token=on_token)
            body, unc = self._split_uncertain(res["text"])
            usage = dict(tokens_in=res["tokens_in"], tokens_out=res["tokens_out"])
            ttft = res["ttft_ms"]
        else:
            body = self._extractive(plan)
            unc = None
            ttft = (time.perf_counter() - t0) * 1000
            if on_token and body:
                on_token(body)
        if missing and not unc:
            unc = "Could not be verified from the corpus: " + "; ".join(missing) + "."
        return self._finish(body, unc, emap, usage, ttft, t0)

    def refine(self, prev, detail, delta_plan, evidence, on_token=None):
        t0 = time.perf_counter()
        emap = {h.chunk.chunk_id: h.chunk for h in evidence}
        missing = [sq for sq, _, ok in delta_plan if not ok]
        usage = dict(tokens_in=0, tokens_out=0)
        if self.llm.available:
            msgs = [{"role": "system", "content": SYS_REFINE},
                    {"role": "user", "content": f"Previous answer (version {prev.version}):\n{prev.answer}\n\n"
                                                f"New detail from the user: {detail}\n\n"
                                                f"Evidence:\n{self._evidence_block(evidence)}"}]
            res = self.llm.complete(msgs, max_tokens=400, on_token=on_token)
            body, unc = self._split_uncertain(res["text"])
            usage = dict(tokens_in=res["tokens_in"], tokens_out=res["tokens_out"])
            ttft = res["ttft_ms"]
        else:
            old = set(split_sentences(prev.answer))
            cand = split_sentences(self._extractive(delta_plan, n_hits=2, n=2))
            fresh = [x for x in cand if x not in old] or cand[:1]  # else point at the rule that applies
            delta = " ".join(fresh)
            body = prev.answer + (f" Update for the new detail: {delta}" if delta else "")
            unc = None
            ttft = (time.perf_counter() - t0) * 1000
            if on_token:
                on_token(body)
        if missing and not unc:
            unc = "The new detail could not be verified from the corpus: " + "; ".join(missing) + "."
        return self._finish(body, unc, emap, usage, ttft, t0)

    def restructure(self, prev, instruction, prev_evidence, on_token=None):
        t0 = time.perf_counter()
        usage = dict(tokens_in=0, tokens_out=0)
        if self.llm.available:
            msgs = [{"role": "system", "content": SYS_FORMAT},
                    {"role": "user", "content": f"Previous answer:\n{prev.answer}\n\nInstruction: {instruction}"}]
            res = self.llm.complete(msgs, max_tokens=250, on_token=on_token)
            body, usage, ttft = res["text"].strip(), dict(tokens_in=res["tokens_in"], tokens_out=res["tokens_out"]), res["ttft_ms"]
        else:
            m = re.search(r"\b(\d+|one|two|three|four|five)\b", instruction.lower())
            n = int(m.group(1)) if m and m.group(1).isdigit() else NUM_WORDS.get(m.group(1), 2) if m else 2
            sents = split_sentences(prev.answer.replace("Update for the new detail:", ""))
            body = "\n".join(f"• {s}" for s in sents[:n])
            ttft = (time.perf_counter() - t0) * 1000
            if on_token:
                on_token(body)
        valid = set(prev.citations)
        body = self._strip_invalid(body, valid)  # no new citations may appear
        out = self._finish(body, prev.uncertainty, prev_evidence, usage, ttft, t0)
        return out

    def _finish(self, body, unc, emap, usage, ttft, t0):
        body = attach_tags(body)
        if not body.strip():
            out = self._finish_raw("", unc, emap, usage, ttft, t0)
            out["answer"] = "I could not verify this from the provided documents."
            return out
        return self._finish_raw(body, unc, emap, usage, ttft, t0)

    def _finish_raw(self, body, unc, emap, usage, ttft, t0):
        rejected = [c for c in extract_citations(body) if c not in emap]
        body = self._strip_invalid(body, set(emap))
        return {"answer": body.strip(), "citations": extract_citations(body), "uncertainty": unc,
                "rejected_citations": rejected, "grounding": self.grounding(body, emap),
                "ttft_ms": round(ttft, 2), "synthesis_ms": round((time.perf_counter() - t0) * 1000, 2), **usage}
