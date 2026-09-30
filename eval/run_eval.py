"""Automated replay benchmark.

Replays every session in the test set through the streaming engine (transcript -> timed chunks),
under the full system, the classic-RAG baseline and ablations, then scores gates G2-G6.

    python -m eval.run_eval                      # all configs
    python -m eval.run_eval --configs full       # just the full system
    python -m eval.run_eval --testset my.json --corpus data/corpus
Outputs: eval/results/results.json, eval/results/report.md, eval/results/turns_<config>.jsonl
"""
import argparse
import json
import os
import statistics
import time
from dataclasses import replace

from app.config import Config
from app.corpus import load_corpus
from app.engine import StreamingRAGEngine
from app.retriever import HybridRetriever

RETRIEVAL_TYPES = {"single", "compound", "late_detail", "out_of_corpus"}
NO_RETRIEVAL_TYPES = {"conversational", "presentation"}
REQUIRED_EVENTS = {"turn_start", "chunk_received", "controller_decision", "utterance_end",
                   "turn_classified", "citation_check", "turn_end"}
TARGETS = {"G2": 0.80, "G3": 0.70, "G4": 0.85, "G6": 1.0}


def doc_of(cid):
    return cid.split(" ")[0]


def pct(x):
    return f"{100 * x:.1f}%" if x is not None else "n/a"


def mean(xs):
    return round(statistics.mean(xs), 2) if xs else None


def p95(xs):
    return round(sorted(xs)[max(0, int(0.95 * len(xs)) - 1)], 2) if xs else None


def run_config(name, cfg, retriever, testset):
    retriever._cache.clear()  # every config starts cold, so latencies are comparable
    eng = StreamingRAGEngine(cfg, retriever=retriever)
    rows = []
    for sess in testset["sessions"]:
        sid = eng.new_session()
        prev = None
        for i, turn in enumerate(sess["turns"]):
            rec = eng.run_utterance(sid, turn["utterance"])
            events = eng.tel[sid].turn_events(rec["turn_id"])
            rows.append({"session": sess["id"], "turn": i + 1, "expected": turn, "record": rec,
                         "events": [e["event"] for e in events],
                         "turn_end": next((e for e in events if e["event"] == "turn_end"), {}),
                         "prev": prev})
            prev = rec
        eng.end_session(sid)
    return rows


def score(rows):
    m, fails = {}, []

    def fail(r, gate, why):
        fails.append({"gate": gate, "session": r["session"], "turn": r["turn"],
                      "utterance": r["expected"]["utterance"], "why": why})

    # G2 early retrieval + false triggers
    elig = [r for r in rows if r["expected"]["type"] in RETRIEVAL_TYPES]
    early = [r for r in elig if r["record"]["metrics"]["early_retrieval"]]
    for r in elig:
        if not r["record"]["metrics"]["early_retrieval"]:
            fail(r, "G2", "retrieval started only after utterance end")
    no_ret = [r for r in rows if r["expected"]["type"] in NO_RETRIEVAL_TYPES]
    false_trig = [r for r in no_ret if r["record"]["retrieval_events"]]
    for r in false_trig:
        fail(r, "G2", "retrieval triggered on a no-retrieval turn")
    m["G2_early_retrieval_rate"] = len(early) / len(elig) if elig else None
    m["false_trigger_rate"] = len(false_trig) / len(no_ret) if no_ret else None
    m["mean_lead_time_s"] = mean([r["record"]["metrics"]["lead_time_s"] for r in early])

    # G3 multi-intent
    comp = [r for r in rows if r["expected"]["type"] == "compound"]
    ok3 = []
    for r in comp:
        n = len(r["record"]["sub_queries"])
        if n >= 2:
            ok3.append(r)
        else:
            fail(r, "G3", f"only {n} sub-query")
    m["G3_multi_intent_rate"] = len(ok3) / len(comp) if comp else None
    m["G3_full_intent_rate"] = (sum(len(r["record"]["sub_queries"]) >= r["expected"].get("min_intents", 2)
                                    for r in comp) / len(comp)) if comp else None

    # Retrieval recall (document level)
    rec_rows = [r for r in rows if r["expected"].get("gold_docs")]
    recalls, cite_recalls = [], []
    for r in rec_rows:
        gold = set(r["expected"]["gold_docs"])
        ev = {doc_of(c) for c in r["record"].get("evidence_ids", [])}
        recalls.append(len(gold & ev) / len(gold))
        if r["expected"]["type"] != "out_of_corpus":
            cite_recalls.append(len(gold & {doc_of(c) for c in r["record"]["citations"]}) / len(gold))
        if gold - ev:
            fail(r, "recall", f"missed gold docs {sorted(gold - ev)}")
    m["retrieval_recall"] = mean(recalls)
    m["citation_recall"] = mean(cite_recalls)

    # G4 grounding
    sents = sup = invalid = 0
    for r in rows:
        g = r["record"]["grounding"]
        sents += g["sentences"]
        sup += g["supported"]
        invalid += len(g["invalid_citations"]) + len(r["record"].get("rejected_citations", []))
        for d in r["record"].get("grounding_details", []):
            if not d["supported"]:
                fail(r, "G4", f"unsupported sentence: {d['sentence'][:90]}")
    m["G4_citation_support"] = sup / sents if sents else None
    m["fabricated_citations"] = invalid
    ooc = [r for r in rows if r["expected"]["type"] == "out_of_corpus"]
    m["uncertainty_on_out_of_corpus"] = (sum(bool(r["record"]["uncertainty"]) for r in ooc) / len(ooc)) if ooc else None
    for r in ooc:
        if not r["record"]["uncertainty"]:
            fail(r, "G4", "no uncertainty flag for out-of-corpus question")
    answerable = [r for r in rows if r["expected"]["type"] in ("single", "compound")]
    m["false_uncertainty_rate"] = (sum(bool(r["record"]["uncertainty"]) for r in answerable) / len(answerable)) if answerable else None

    # G5 session refinement
    late = [r for r in rows if r["expected"]["type"] == "late_detail"]
    ok5 = []
    for r in late:
        rec, prev = r["record"], r["prev"]
        why = []
        if rec["turn_type"] != "late_detail":
            why.append(f"classified as {rec['turn_type']}")
        if not prev or rec.get("parent_version") != prev.get("answer_version"):
            why.append("version lineage broken")
        if not (rec.get("citation_lineage") or {}).get("kept"):
            why.append("prior citations not preserved")
        prev_q = set(prev["sub_queries"]) if prev else set()
        if any(e["query"] in prev_q for e in rec["retrieval_events"]):
            why.append("re-ran full previous search")
        if why:
            fail(r, "G5", "; ".join(why))
        else:
            ok5.append(r)
    m["G5_refinement_pass_rate"] = len(ok5) / len(late) if late else None

    # presentation suppression correctness
    pres = [r for r in rows if r["expected"]["type"] == "presentation"]
    okp = [r for r in pres if r["record"]["turn_type"] == "presentation" and not r["record"]["retrieval_events"]
           and set(r["record"]["citations"]) <= set((r["prev"] or {}).get("citations", []))]
    for r in pres:
        if r not in okp:
            fail(r, "suppression", f"turn_type={r['record']['turn_type']}, retrievals={len(r['record']['retrieval_events'])}")
    m["presentation_suppression_rate"] = len(okp) / len(pres) if pres else None

    # G6 telemetry coverage
    ok6 = 0
    for r in rows:
        need = set(REQUIRED_EVENTS)
        if r["record"]["retrieval_events"]:
            need |= {"retrieval_started", "retrieval_completed"}
        if r["record"]["turn_type"] in ("new_query", "late_detail"):
            need.add("answer_version")
        te = r["turn_end"]
        if need <= set(r["events"]) and all(k in te for k in ("tokens_in", "tokens_out", "cost_usd", "ttft_ms")):
            ok6 += 1
        else:
            fail(r, "G6", f"missing events {sorted(need - set(r['events']))}")
    m["G6_trace_coverage"] = ok6 / len(rows) if rows else None

    # latency / cost
    mets = [r["record"]["metrics"] for r in rows]
    m["ttft_ms_mean"] = mean([x["ttft_ms"] for x in mets])
    m["ttft_ms_p95"] = p95([x["ttft_ms"] for x in mets])
    m["post_utterance_latency_ms_mean"] = mean([x["total_ms_after_end"] for x in mets])
    lat = [e["latency_ms"] for r in rows for e in r["record"]["retrieval_events"]]
    m["retrieval_latency_ms_mean"] = mean(lat)
    m["retrieval_calls_per_turn"] = mean([x["retrieval_calls"] for x in mets])
    m["tokens_per_turn"] = mean([x["tokens_in"] + x["tokens_out"] for x in mets])
    m["cost_usd_per_turn"] = round(statistics.mean([x["cost_usd"] for x in mets]), 6) if mets else None
    m["turns"] = len(rows)
    return m, fails


def build_configs(base: Config, have_dense: bool, have_llm: bool):
    cfgs = {
        "full": base,
        "baseline_classic_rag": replace(base, streaming=False, decomposition=False, refinement=False, suppression=False),
        "ablation_no_decomposition": replace(base, decomposition=False),
        "ablation_sparse_only": replace(base, retrieval_mode="sparse"),
    }
    if have_dense:
        cfgs["ablation_dense_only"] = replace(base, retrieval_mode="dense")
    if have_llm:
        cfgs["ablation_llm_decomposer"] = replace(base, decomposer_mode="llm")
    return cfgs


def write_report(path, results, index_info, failures_full):
    L = ["# Streaming Live RAG - Evaluation Report", "",
         f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}  ",
         f"Corpus: {index_info['chunks']} chunks from `{index_info['corpus']}`; "
         f"indexing time {index_info['index_ms']} ms (retrieval mode available: {index_info['mode']}); "
         f"LLM: {index_info['llm']}", ""]
    full = results.get("full", {})
    L += ["## Acceptance gates (full system)", "", "| Gate | Criterion | Target | Result | Pass |", "|---|---|---|---|---|"]
    gates = [
        ("G1", "Reproducibility", "Pass/Fail", "replay suite completed", True),
        ("G2", "Early retrieval", ">= 80%", pct(full.get("G2_early_retrieval_rate")),
         (full.get("G2_early_retrieval_rate") or 0) >= TARGETS["G2"]),
        ("G3", "Multi-intent identification", ">= 70%", pct(full.get("G3_multi_intent_rate")),
         (full.get("G3_multi_intent_rate") or 0) >= TARGETS["G3"]),
        ("G4", "Factual grounding", ">= 85% citation support", pct(full.get("G4_citation_support")),
         (full.get("G4_citation_support") or 0) >= TARGETS["G4"] and full.get("fabricated_citations", 1) == 0),
        ("G5", "Session refinement", "state continuity", pct(full.get("G5_refinement_pass_rate")),
         (full.get("G5_refinement_pass_rate") or 0) >= 0.999),
        ("G6", "Telemetry", "100% trace coverage", pct(full.get("G6_trace_coverage")),
         (full.get("G6_trace_coverage") or 0) >= 0.999),
    ]
    for g, c, t, res, ok in gates:
        L.append(f"| {g} | {c} | {t} | {res} | {'yes' if ok else 'no'} |")
    keys = ["G2_early_retrieval_rate", "false_trigger_rate", "mean_lead_time_s", "G3_multi_intent_rate",
            "G3_full_intent_rate", "retrieval_recall", "citation_recall", "G4_citation_support",
            "fabricated_citations", "uncertainty_on_out_of_corpus", "false_uncertainty_rate",
            "G5_refinement_pass_rate", "presentation_suppression_rate", "G6_trace_coverage", "ttft_ms_mean",
            "ttft_ms_p95", "post_utterance_latency_ms_mean", "retrieval_latency_ms_mean",
            "retrieval_calls_per_turn", "tokens_per_turn", "cost_usd_per_turn"]
    names = list(results)
    L += ["", "## Full system vs baseline vs ablations", "", "| Metric | " + " | ".join(names) + " |",
          "|---|" + "---|" * len(names)]
    for k in keys:
        cells = []
        for n in names:
            v = results[n].get(k)
            cells.append(pct(v) if isinstance(v, float) and ("rate" in k or "recall" in k or "support" in k
                                                             or "coverage" in k or "uncertainty" in k) else str(v))
        L.append(f"| {k} | " + " | ".join(cells) + " |")
    L += ["", "## Failed checks in the full system (use for edge-case analysis)", ""]
    if not failures_full:
        L.append("None.")
    for f in failures_full:
        L.append(f"- **{f['gate']}** {f['session']} turn {f['turn']} - \"{f['utterance']}\": {f['why']}")
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(L) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--testset", default="eval/testset.json")
    ap.add_argument("--corpus", default=None)
    ap.add_argument("--out", default="eval/results")
    ap.add_argument("--configs", nargs="*", default=None)
    a = ap.parse_args()
    base = Config()
    if a.corpus:
        base.corpus_dir = a.corpus
    base.log_dir = os.path.join(a.out, "logs")
    os.makedirs(a.out, exist_ok=True)
    testset = json.load(open(a.testset, encoding="utf-8"))

    chunks = load_corpus(base.corpus_dir)
    t0 = time.perf_counter()
    hybrid = HybridRetriever(chunks, base.retrieval_mode, base.embed_model)
    index_ms = round((time.perf_counter() - t0) * 1000, 1)
    have_dense = hybrid.dense is not None
    retrievers = {base.retrieval_mode: hybrid, "sparse": HybridRetriever(chunks, "sparse")}
    if have_dense:
        dense = HybridRetriever.__new__(HybridRetriever)
        dense.__dict__.update(hybrid.__dict__)
        dense.mode, dense._cache = "dense", {}
        retrievers["dense"] = dense
    have_llm = bool(base.llm_api_key)

    cfgs = build_configs(base, have_dense, have_llm)
    if a.configs:
        cfgs = {k: v for k, v in cfgs.items() if k in a.configs}
    results, all_fails = {}, {}
    for name, cfg in cfgs.items():
        t = time.perf_counter()
        rows = run_config(name, cfg, retrievers[cfg.retrieval_mode], testset)
        results[name], all_fails[name] = score(rows)
        results[name]["runtime_s"] = round(time.perf_counter() - t, 2)
        with open(os.path.join(a.out, f"turns_{name}.jsonl"), "w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps({"session": r["session"], "turn": r["turn"], "expected": r["expected"],
                                     "record": {k: v for k, v in r["record"].items() if k != "grounding_details"}},
                                    ensure_ascii=False) + "\n")
        print(f"[{name}] G2={pct(results[name]['G2_early_retrieval_rate'])} "
              f"G3={pct(results[name]['G3_multi_intent_rate'])} G4={pct(results[name]['G4_citation_support'])} "
              f"G5={pct(results[name]['G5_refinement_pass_rate'])} G6={pct(results[name]['G6_trace_coverage'])} "
              f"recall={results[name]['retrieval_recall']} ttft={results[name]['ttft_ms_mean']}ms")
    info = {"chunks": len(chunks), "corpus": base.corpus_dir, "index_ms": index_ms,
            "mode": hybrid.mode, "llm": base.llm_model if have_llm else "none (offline extractive mode)"}
    json.dump({"info": info, "results": results, "failures": all_fails},
              open(os.path.join(a.out, "results.json"), "w", encoding="utf-8"), indent=2, ensure_ascii=False)
    write_report(os.path.join(a.out, "report.md"), results, info, all_fails.get("full", []))
    print(f"Report written to {os.path.join(a.out, 'report.md')}")


if __name__ == "__main__":
    main()
