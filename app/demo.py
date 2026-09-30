"""Terminal demo: type what the user "says"; it is streamed chunk-by-chunk in (simulated) real time
and every controller decision, retrieval, answer token, citation and metric is shown.

    python -m app.demo              # real-time pacing
    python -m app.demo --fast       # no pacing
    python -m app.demo --script eval/demo_script.txt   # replay a scripted conversation

Commands:  :new  (new session)   :log  (last turn's telemetry)   :json  (last output record)   :quit
"""
import argparse
import json
import sys
import time

from .engine import StreamingRAGEngine

C = {"wait": "\033[90m", "retrieve": "\033[92m", "suppress": "\033[95m", "head": "\033[96m",
     "warn": "\033[93m", "end": "\033[0m", "bold": "\033[1m"}


def color(k, s):
    return f"{C[k]}{s}{C['end']}" if sys.stdout.isatty() else s


def show_chunk(out):
    tag = color(out["decision"], f"{out['decision'].upper():9}")
    extra = f"  -> {out['dispatched']}" if out["dispatched"] else ""
    print(f"  [{out['t']:5.2f}s] {out['chunk']!r:38} {tag} {out['reason']}{extra}")


def run_turn(eng, sid, text, realtime):
    print(color("head", "  stream:"))
    rec = eng.run_utterance(sid, text, realtime=realtime, on_chunk=show_chunk)
    m = rec["metrics"]
    print(color("head", f"  [{m['utterance_end_s']:5.2f}s] [utterance end]  turn_type={rec['turn_type']}  "
                        f"retrieval_required={rec['retrieval_required']} ({rec['reason']})"))
    if rec["sub_queries"]:
        print(color("head", "  sub-queries: ") + " | ".join(rec["sub_queries"]))
    ver = f"v{rec['answer_version']}" if rec["answer_version"] else "-"
    print(color("bold", f"\n  ANSWER ({ver}):"))
    print("  " + rec["answer"].replace("\n", "\n  "))
    print(color("head", "  citations: ") + (", ".join(rec["citations"]) or "none"))
    if rec["uncertainty"]:
        print(color("warn", "  uncertainty: " + rec["uncertainty"]))
    if rec["citation_lineage"]:
        print(color("head", "  lineage: ") + json.dumps(rec["citation_lineage"], ensure_ascii=False))
    g = rec["grounding"]
    early = f"yes, {m['lead_time_s']}s before end" if m["early_retrieval"] else "no"
    print(color("head", "  metrics: ") + f"early retrieval={early} | retrieval calls={m['retrieval_calls']} | "
          f"TTFT after end={m['ttft_ms']} ms | grounded sentences={g['supported']}/{g['sentences']} | "
          f"tokens={m['tokens_in']}+{m['tokens_out']} | cost=${m['cost_usd']}\n")
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fast", action="store_true")
    ap.add_argument("--script", help="text file, one utterance per line; blank line = new session")
    a = ap.parse_args()
    eng = StreamingRAGEngine()
    print(color("bold", f"Streaming Live RAG demo | {len(eng.retriever.chunks)} chunks | retrieval={eng.retriever.mode} | "
                        f"LLM={'%s' % eng.cfg.llm_model if eng.llm.available else 'offline extractive'}"))
    sid, last = eng.new_session(), None
    lines = open(a.script, encoding="utf-8").read().splitlines() if a.script else None
    while True:
        if lines is not None:
            if not lines:
                break
            text = lines.pop(0).strip()
            if not text:
                sid = eng.new_session()
                print(color("head", "---- new session ----"))
                continue
            print(color("bold", f"USER> {text}"))
            time.sleep(0.4)
        else:
            try:
                text = input(color("bold", "USER> ")).strip()
            except (EOFError, KeyboardInterrupt):
                break
        if not text:
            continue
        if text == ":quit":
            break
        if text == ":new":
            sid = eng.new_session()
            print(color("head", "---- new session ----"))
            continue
        if text == ":log":
            for ev in eng.tel[sid].turn_events(eng.store.get(sid).turn_count):
                print("  " + json.dumps({k: v for k, v in ev.items() if k not in ("session_id", "ts")}, ensure_ascii=False))
            continue
        if text == ":json" and last:
            print(json.dumps({k: v for k, v in last.items() if k not in ("timeline", "grounding_details")},
                             indent=2, ensure_ascii=False))
            continue
        last = run_turn(eng, sid, text, realtime=not a.fast)


if __name__ == "__main__":
    main()
