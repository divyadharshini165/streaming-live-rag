# Streaming Live RAG

**Samsung PRISM GenAI Hackathon 2026 · Theme 04: Streaming Live RAG**

An event-driven RAG engine for full-duplex conversation. It listens to a transcript **chunk by chunk**, starts retrieving **before the user finishes speaking**, splits one spoken request into the **several queries it implies**, **refines the previous answer** when a late detail arrives (instead of restarting), **skips retrieval** for reformat or small-talk turns, and cites a corpus chunk for **every fact**. If the corpus lacks the answer, it says so explicitly.

---
## Submission Deliverables

- **Demo Video (YouTube):** [Watch 5-Min Video](https://youtu.be/AaopxFCN7rw?si=9Xv12n1SnxNEPGe3)
- **Presentation Deck:** [`docs/SRMIST_Live RAGrets_Submission.pptx`](./docs/SRMIST_Live%20RAGrets_Submission.pptx)
- **AI Usage Disclosure Form:** [`docs/SRMIST_Live RAGrets_AI_Disclosure.docx`](./docs/SRMIST_Live%20RAGrets_AI_Disclosure.docx)
- **Git Release Tag:** `PRISM_GENAI_HACKATHON_Y2026`

## Quick start

### Option A: Docker (one command)

```bash
docker compose up --build
```

This builds the image (CPU-only), **runs the full automated replay benchmark** (report written to `eval/results/report.md`), then serves the API on `http://localhost:8000`. No manual steps and no API key are needed.

### Option B: Local Python (3.11+)

```bash
python -m venv .venv && source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
cp .env.example .env                                     # optional: add an LLM key

python -m app.demo                                       # interactive live demo
python -m eval.run_eval                                  # benchmark + report
uvicorn app.api:app --port 8000                          # API server
pytest                                                   # tests
```

### LLM (optional)

The system runs **fully offline** by default, using rule-based decomposition and extractive, sentence-level cited answers. To get fluent synthesized answers, put any OpenAI-compatible key in `.env`: Groq's free tier is the default, and Gemini or a local Ollama model also work (examples are in `.env.example`). The LLM is used for at most **one call per turn** (answer synthesis), plus optionally one for decomposition (`DECOMPOSER_MODE=llm`).

---

## Architecture

```
 transcript chunks ──► [1] Retrieval Controller ──► [2] Multi-Intent Decomposer ──► [3] Hybrid Retrieval & Fusion ──► [4] Session-Aware Synthesis ──► answer + citations
 (t=1.4s, 2.9s, …)      wait / retrieve /            clause split + context          BM25 + bge-small dense           grounded answer, citation
                        suppress, per chunk          carry + dedupe, parallel        RRF fusion, coverage rerank,     check, uncertainty flag,
                        turn type at end             dispatch mid-utterance          dedupe, session-topic boost      answer versioning (v1→v2)
                                     │                          │                              │                              │
                                     └──────────────────────────┴──────── [5] Telemetry (JSONL events per session) ───────────┘
```

| # | Component | File | What it does |
|---|---|---|---|
| 1 | Retrieval Controller | `app/controller.py` | For every chunk: **WAIT** (intent unstable, e.g. sentence ends in "in…"), **RETRIEVE** (enough informative corpus terms), or **SUPPRESS** (reformat request). At utterance end, classifies the turn as `new_query`, `late_detail`, `presentation`, or `conversational`. |
| 2 | Multi-Intent Decomposer | `app/decomposer.py` | Splits at clause boundaries, carries shared context (place, group size, topic) into short clauses, turns group sizes into capacity queries, and drops near-duplicates to prevent over-fragmentation. Completed clauses are dispatched **while the user is still speaking**. Optional LLM mode. |
| 3 | Evidence Fusion & Reranking | `app/retriever.py` | Hybrid **BM25 + dense** (`BAAI/bge-small-en-v1.5`, CPU), **Reciprocal Rank Fusion**, lexical-coverage rerank, and near-duplicate removal. Across sub-queries, each sub-query's best hits are guaranteed, then the rest are RRF-ordered. |
| 4 | Session-Only Refinement | `app/engine.py`, `app/session.py`, `app/synthesizer.py` | Late details become **delta queries** (only the new constraint plus the prior topic), with a small boost for documents already used in this session. The previous answer is kept and updated into a new version, and citation lineage (`kept / added / dropped`) is recorded. Memory is in-process and per session only. |
| 5 | Grounding & Uncertainty | `app/synthesizer.py` | IDF-weighted sufficiency check per sub-query leads to an explicit `uncertainty` field. Each answer sentence must cite a retrieved chunk that contains its terms, and fabricated tags are stripped and logged. |
| 6 | Observability | `app/telemetry.py` | Structured JSONL events with stream timestamps, wall-clock latency, decisions, queries, citations, versions, tokens, and cost. |

**Why this design (architectural parsimony):** there's no agent framework. The only model calls are the embedding model and, optionally, one LLM call per turn. The controller and decomposer are rule-based and run in under 1 ms, so early retrieval costs almost nothing. Ablations below show what each part adds.

---

## Try it

### Terminal demo (used for the demo video)

```bash
python -m app.demo                                   # type utterances; streamed at speaking pace
python -m app.demo --fast --script eval/demo_script.txt
```

Sample output:

```
USER> I need to plan a customer workshop in Pune for 30 people, and I need the cancellation policy and the catering options.
  [ 1.43s] 'I need to plan'                 WAIT      no_retrievable_intent_yet
  [ 2.86s] 'a customer workshop in'         WAIT      intent_unstable
  [ 4.29s] 'Pune for 30 people,'            RETRIEVE  intent_stable  -> ['customer workshop pune 30 people']
  [ 7.14s] 'cancellation policy and the'    RETRIEVE  multi_intent_detected  -> ['Pune 30 people venue capacity seats', 'cancellation policy Pune']
  [ 7.86s] 'catering options.'              RETRIEVE  multi_intent_detected  -> ['catering Pune workshop']
  [ 8.16s] [utterance end]  turn_type=new_query
```

Demo commands: `:new` (new session), `:log` (telemetry of the last turn), `:json` (full output record), `:quit`.

### REST API

```bash
SID=$(curl -s -X POST localhost:8000/sessions | python -c "import sys,json;print(json.load(sys.stdin)['session_id'])")

# stream chunks as they are transcribed
curl -X POST localhost:8000/sessions/$SID/chunks -H 'Content-Type: application/json' -d '{"text":"What is the per","t":1.4}'
curl -X POST localhost:8000/sessions/$SID/chunks -H 'Content-Type: application/json' -d '{"text":"diem for international","t":2.8}'
curl -X POST localhost:8000/sessions/$SID/chunks -H 'Content-Type: application/json' -d '{"text":"travel?","t":3.2}'
curl -X POST localhost:8000/sessions/$SID/end    -H 'Content-Type: application/json' -d '{"t":3.5}'

# or replay a whole utterance as timed chunks
curl -X POST localhost:8000/sessions/$SID/utterance -H 'Content-Type: application/json' -d '{"text":"Make that shorter."}'

curl localhost:8000/sessions/$SID/telemetry
```

| Method | Path | Purpose |
|---|---|---|
| POST | `/sessions` | Create a session |
| POST | `/sessions/{id}/chunks` `{text, t}` | Send one transcript chunk; returns the controller decision |
| POST | `/sessions/{id}/end` `{t}` | Utterance ended; returns the output record |
| POST | `/sessions/{id}/utterance` `{text, realtime?}` | Replay a full utterance as timed chunks |
| GET | `/sessions/{id}/telemetry` | All telemetry events of the session |
| DELETE | `/sessions/{id}` | Drop session memory |
| GET | `/health` | Corpus size, retrieval mode, LLM, indexing time |
| WS | `/ws/{id}` | Send `{"type":"chunk","text","t"}` / `{"type":"end","t"}`; receive `decision`, streamed `token`, `final` |

### Output record

```json
{
  "turn_type": "new_query",
  "retrieval_required": true,
  "reason": "retrieval_required",
  "retrieval_events": [
    {"timestamp_s": 4.29, "query": "customer workshop pune 30 people", "trigger": "provisional", "latency_ms": 1.2, "top_chunks": ["Doc_06 §1", "Doc_06 §3", "Doc_07 §2"]},
    {"timestamp_s": 7.14, "query": "cancellation policy Pune", "trigger": "multi_intent", "latency_ms": 1.1, "top_chunks": ["Doc_08 §4", "Doc_07 §4", "Doc_09 §4"]}
  ],
  "sub_queries": ["Pune 30 people venue capacity seats", "cancellation policy Pune", "catering Pune workshop"],
  "answer": "The Training Room seats 35 people in classroom style or 40 in theatre style [Doc_07 §2]. …",
  "citations": ["Doc_07 §2", "Doc_08 §4", "Doc_06 §5"],
  "uncertainty": null,
  "answer_version": 1,
  "parent_version": null,
  "citation_lineage": null,
  "grounding": {"sentences": 8, "supported": 8, "support_rate": 1.0, "invalid_citations": []},
  "metrics": {"utterance_end_s": 8.16, "first_retrieval_s": 4.29, "early_retrieval": true, "lead_time_s": 3.87,
              "ttft_ms": 1.8, "retrieval_calls": 4, "tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0}
}
```

---

## Using a different corpus

Put `.md`, `.txt`, or `.pdf` files in `data/corpus/` (or set `CORPUS_DIR`). The corpus is indexed at startup, and nothing in the code depends on specific documents.

- Files named `Doc_<n>_*.md` keep that ID. Other files get IDs in sorted order.
- Markdown `##` headings become citable sections (`Doc_07 §4`). Files without headings are split into roughly 180-word windows (`Doc_03 §1`, `§2`, …).
- With Docker, the corpus folder is mounted, so swapping documents needs no rebuild.

The included `data/corpus/` is a **sample corpus we wrote** (fictional company "Nexora Technologies": travel policy, venues, caterers, HR policies), because no corpus was shared for Theme 4.

---

## Evaluation

```bash
python -m eval.run_eval                              # all configs → eval/results/report.md + results.json
python -m eval.run_eval --configs full               # full system only
python -m eval.run_eval --testset my_tests.json --corpus path/to/corpus
```

`eval/testset.json` has 22 sessions and 32 turns: single, compound, late-detail, presentation, conversational, and out-of-corpus. Each session is replayed chunk by chunk under the full system, a **classic-RAG baseline** (waits for the utterance end, one query, no refinement, no suppression), and **ablations** (no decomposition, sparse-only, dense-only, and an LLM decomposer when a key is set).

The following results come from a run in **offline mode with BM25 only** (the dense model was not available in the build sandbox). Regenerate them with `docker compose up`, which uses hybrid retrieval.

| Gate | Criterion | Target | Full system | Baseline |
|---|---|---|---|---|
| G2 | Early retrieval (before utterance end) | ≥ 80% | 100% (0% false triggers) | 0% (100% false triggers) |
| G3 | Multi-intent identification | ≥ 70% | 100% | 0% |
| G4 | Citation support / fabricated IDs | ≥ 85% / 0 | 100% / 0 | 100% / 0 |
| G5 | Refinement with state continuity | pass | 100% | 0% |
| G6 | Telemetry trace coverage | 100% | 100% | 100% |
| – | Citation recall (gold docs cited) | – | 96% | 70% |
| – | False "uncertain" on answerable turns | – | 5.3% | 15.8% |

> **Honest note:** the test set and corpus were written by the team, so these numbers are optimistic. Samsung's held-out set will be harder. `report.md` lists every failed check for edge-case analysis.

---

## Telemetry schema

Each session writes `logs/<session_id>.jsonl`, one event per line. Every event includes these common fields:

```
ts          unix time            session_id, turn_id
event       event name           stream_t_s  position in the audio stream (s)
wall_ms     processing time since the turn started
```

| Event | Extra fields |
|---|---|
| `session_start` | corpus_chunks, retrieval_mode, llm, config |
| `turn_start` | has_prior_answer |
| `chunk_received` | text |
| `controller_decision` | decision (`wait`/`retrieve`/`suppress`), reason, dispatched |
| `retrieval_started` | query, trigger (`provisional`/`multi_intent`/`late_detail`/`final`) |
| `retrieval_completed` | query, trigger, latency_ms, top_chunks, hits[id, score, coverage] |
| `utterance_end` | text |
| `turn_classified` | turn_type, retrieval_required |
| `decomposition` | mode, sub_queries |
| `sufficiency_check` | per sub-query: supported |
| `fusion` | candidates / prior_evidence / delta_evidence, evidence ids |
| `first_token` | ttft_ms (after utterance end) |
| `answer_version` | version, parent, turn_type, lineage {kept, added, dropped} |
| `citation_check` | citations, rejected, support_rate, sentences |
| `turn_end` | utterance_end_s, first_retrieval_s, early_retrieval, lead_time_s, ttft_ms, total_ms_after_end, retrieval_calls, tokens_in, tokens_out, cost_usd |

---

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `CORPUS_DIR` | `data/corpus` | Folder of documents |
| `RETRIEVAL_MODE` | `hybrid` | `hybrid`, `sparse`, `dense` |
| `EMBED_MODEL` | `BAAI/bge-small-en-v1.5` | Dense embedding model (CPU) |
| `TOP_K` / `EVIDENCE_CAP` | 5 / 8 | Hits per query / chunks sent to synthesis |
| `DECOMPOSER_MODE` | `rule` | `rule` or `llm` |
| `LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL` | empty / Groq / `llama-3.1-8b-instant` | Any OpenAI-compatible endpoint |
| `PRICE_IN_PER_M_USD`, `PRICE_OUT_PER_M_USD` | 0.05 / 0.08 | For cost-per-turn estimates; set to your provider's price |
| `WORDS_PER_SECOND`, `WORDS_PER_CHUNK` | 2.8 / 4 | Simulated speech pacing |

---

## Compliance with the theme rules

- **Corpus isolation:** answers come only from retrieved chunks. The LLM is instructed to use only the evidence, and every tag is validated against the retrieved set.
- **No hardcoding:** controller and decomposer rules are generic language cues plus statistics computed from whatever corpus is loaded. No test prompts, queries, or answers exist in application code, and the test set is used only by `eval/`.
- **Grounding:** every sentence is checked against the chunk it cites. Unsupported sub-questions produce an explicit `uncertainty`.
- **Session-bound state:** memory lives in process, per session, with a TTL. Nothing is persisted across sessions except logs.
- **Parsimony:** no agent framework, and at most one LLM call per turn.

## Known limitations

- **Offline mode** can point to the right section but cannot do arithmetic over it. For example, "cancel 10 days before" returns the correct cancellation clause but not the specific tier. LLM mode resolves this.
- The rule-based controller and decomposer are tuned for English and for clause-structured speech. Very long run-on utterances may under-split (the LLM decomposer is available as a fallback).
- The sufficiency check is lexical (IDF-weighted). Paraphrases with no word overlap can be flagged uncertain in offline mode.

## Project structure

```
app/
  config.py        settings from environment
  corpus.py        loader + section chunker (.md/.txt/.pdf)
  text.py          tokenizer, stemmer, sentence splitter, citation parsing
  retriever.py     BM25 + dense + RRF + rerank + dedupe
  controller.py    wait / retrieve / suppress + turn classification
  decomposer.py    multi-intent + late-detail delta queries
  synthesizer.py   grounded answer, refinement, restructure, citation check
  session.py       ephemeral session memory + answer versions
  telemetry.py     JSONL observability events
  engine.py        event-driven orchestration
  simulator.py     transcript -> timed chunks
  api.py           FastAPI REST + WebSocket
  demo.py          terminal demo
data/corpus/       sample corpus (18 documents)
eval/              test set, replay benchmark, demo script, results
tests/             pytest suite (includes a fake-LLM test)
```

## Submission

```bash
git tag PRISM_GENAI_HACKATHON_Y2026
git push origin PRISM_GENAI_HACKATHON_Y2026
```
