# Streaming Live RAG - Evaluation Report

Generated: 2026-09-30 12:24:00  
Corpus: 76 chunks from `/app/data/corpus`; indexing time 21761.7 ms (retrieval mode available: hybrid); LLM: none (offline extractive mode)

## Acceptance gates (full system)

| Gate | Criterion | Target | Result | Pass |
|---|---|---|---|---|
| G1 | Reproducibility | Pass/Fail | replay suite completed | yes |
| G2 | Early retrieval | >= 80% | 100.0% | yes |
| G3 | Multi-intent identification | >= 70% | 100.0% | yes |
| G4 | Factual grounding | >= 85% citation support | 100.0% | yes |
| G5 | Session refinement | state continuity | 100.0% | yes |
| G6 | Telemetry | 100% trace coverage | 100.0% | yes |

## Full system vs baseline vs ablations

| Metric | full | baseline_classic_rag | ablation_no_decomposition | ablation_sparse_only | ablation_dense_only |
|---|---|---|---|---|---|
| G2_early_retrieval_rate | 100.0% | 0.0% | 100.0% | 100.0% | 100.0% |
| false_trigger_rate | 0.0% | 100.0% | 0.0% | 0.0% | 0.0% |
| mean_lead_time_s | 1.65 | None | 1.65 | 1.65 | 1.65 |
| G3_multi_intent_rate | 100.0% | 0.0% | 0.0% | 100.0% | 100.0% |
| G3_full_intent_rate | 100.0% | 0.0% | 0.0% | 100.0% | 100.0% |
| retrieval_recall | 100.0% | 96.0% | 100.0% | 100.0% | 100.0% |
| citation_recall | 93.0% | 70.0% | 78.0% | 96.0% | 93.0% |
| G4_citation_support | 100.0% | 100.0% | 98.1% | 100.0% | 100.0% |
| fabricated_citations | 0 | 0 | 0 | 0 | 0 |
| uncertainty_on_out_of_corpus | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| false_uncertainty_rate | 5.3% | 15.8% | 15.8% | 5.3% | 5.3% |
| G5_refinement_pass_rate | 100.0% | 0.0% | 75.0% | 100.0% | 100.0% |
| presentation_suppression_rate | 100.0% | 0.0% | 100.0% | 100.0% | 100.0% |
| G6_trace_coverage | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| ttft_ms_mean | 158.22 | 247.28 | 150.86 | 115.15 | 174.79 |
| ttft_ms_p95 | 379.38 | 314.31 | 284.98 | 210.83 | 465.72 |
| post_utterance_latency_ms_mean | 225.1 | 297.49 | 214.72 | 185.15 | 243.45 |
| retrieval_latency_ms_mean | 115.83 | 70.73 | 74.7 | 3.76 | 134.73 |
| retrieval_calls_per_turn | 1.75 | 1 | 1.28 | 1.75 | 1.75 |
| tokens_per_turn | 0 | 0 | 0 | 0 | 0 |
| cost_usd_per_turn | 0.0 | 0.0 | 0.0 | 0.0 | 0.0 |

## Failed checks in the full system (use for edge-case analysis)

None.
