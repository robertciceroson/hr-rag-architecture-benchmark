# HR RAG Architecture Benchmark

*Which RAG design actually works for HR policy Q&A? Measured, not assumed.*

![CI](https://github.com/robertciceroson/hr-rag-architecture-benchmark/actions/workflows/ci.yml/badge.svg)

This is the third project in a series:

| Repo | Role |
|---|---|
| [HR-Policy-QA-Bot](https://github.com/robertciceroson/HR-Policy-QA-Bot) | **Build**: LangChain + FAISS + FastEmbed + Groq (gpt-oss-120b) RAG chatbot answers...|
| [hr-policy-eval-harness](https://github.com/robertciceroson/hr-policy-eval-harness) | **Evaluate**: 73-case golden dataset; baseline scored 87.7% and exposed a 30% out-of-scope detection rate |
| **hr-rag-architecture-benchmark** (this repo) | **Improve**: 10 controlled experiments to find which architecture fixes those failures, and at what cost |

The experiments follow the RAG, chunking, HyDE, and agentic-RAG chapters of
the *AI Engineering Guidebook* (Daily Dose of Data Science, 2025). The full
design, hypotheses, and decision rules are in **[EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)**.

## What's being compared

| ID | Architecture |
|---|---|
| E0 | TF-IDF, section chunks (original baseline, control) |
| E1 | BM25 |
| E2 | Dense embeddings (bge-small, same as production bot) |
| E3 | Hybrid: BM25 + dense via Reciprocal Rank Fusion |
| E4 | Dense, fixed-size chunks |
| E5 | Dense, sentence-window chunks |
| E6 | Dense, section chunks with document-title headers |
| E7 | Hybrid + cross-encoder reranker |
| E8 | HyDE (LLM-generated hypothetical answer) + dense |
| E9 | Dense + LLM scope gate |
| E10 | Dense + cross-encoder reranker (ablation of E7: is BM25 needed?) |
| E11 | Dense + reranker + LLM scope gate |

## How it's kept fair

- **One variable per experiment.** The dataset, documents, guardrail rules, and scoring rules never change. Adversarial items must stay 10/10 in every run, and a test enforces this.
- **Per-retriever calibration.** Each retriever gets its own scope and ambiguity thresholds, because TF-IDF, BM25, embedding, and cross-encoder scores live on different scales.
- **Cross-validated headline.** Thresholds are tuned on 4/5 of the items and scored on the held-out 1/5. Tuning on the test set is what inflated the original 87.7%.
- **Significance testing.** An exact McNemar test compares each experiment with the baseline on the same 73 items. With n = 73, a gain of one or two items is noise and gets reported as noise.

## Findings

All 12 experiments complete (E0–E11). LLM experiments use `openai/gpt-oss-120b` on Groq.

| | CV accuracy | vs honest baseline | Out-of-scope recall / precision | Cost per question |
|---|---|---|---|---|
| Baseline as published (tuned on test set) | 64/73 (87.7%) | — | 30% / — | <1 ms |
| **Baseline, honest (cross-validated)** | **61/73 (83.6%)** | — | 30% / 50% | <1 ms |
| Dense embeddings (E2) | 65/73 (89.0%) | +10/−6, p = 0.45 (n.s.) | 80% / 62% | ~20 ms |
| Dense + cross-encoder rerank (E10) | 69/73 (94.5%) | +10/−2, p = 0.04 | 90% / 75% | ~1.3 s CPU, no API |
| Dense + LLM scope gate (E9) | 70/73 (95.9%) | +11/−2, p = 0.02 | 90% / 90% | ~10 ms + 1 LLM call |
| **Dense + rerank + LLM scope gate (E11)** | **72/73 (98.6%)** | **+12/−1, p = 0.003** | **100% / 91%** | ~1 s CPU + 1 LLM call |

1. **The published 87.7% was optimistic.** Its threshold was tuned on the same 73 items it was scored on. Honest cross-validated calibration puts the baseline at 83.6%, and every experiment is judged against that.
2. **Scoping, not ranking, was the real problem.** Dense retrieval already found the right policy for 44 of 45 answerable questions. What failed was deciding when a question is *not* covered. The two components that fixed it read the question and the text together: a cross-encoder reranker and an LLM scope check.
3. **Best result: reranker + LLM gate (E11), 98.6%.** Every answerable question is answered from the right policy, and all 10 uncovered topics are refused, including "tuition reimbursement," which no similarity score could separate from "expense reimbursement." Calibration effectively handed scoping to the LLM and ranking to the reranker.
4. **But the top three are statistically tied.** At n = 73, E9, E10, and E11 are not distinguishable from each other (E11 vs E10: p = 0.38). The choice between them is a cost and data-handling decision, not an accuracy one:
   - **E11** for maximum accuracy, if sending policy excerpts *and employee questions* to an external LLM API is acceptable. HR questions can contain personal details.
   - **E10** if data must stay in-house: no external API, 94.5%, about 1.3 s per question on CPU.
   - **E9** for the lightest footprint: no reranker, fast retrieval, one small LLM call.
5. **Two popular "improvements" hurt, and one prediction was wrong.** Hybrid BM25 fusion (E3) and document-title chunk headers (E6) both made scope detection worse. HyDE (E8) was predicted to hurt scope detection; instead it helped (precision 62% → 89%) and fixed ambiguity handling.
6. **Limits:** 73 items and 10 out-of-scope cases, a 6-document synthetic corpus (the reranker sees 20 of 36 chunks, so "BM25 adds nothing" is a small-corpus result), one experiment that degenerated (E5), and LLM results specific to one model. Details in [EXPERIMENT_PLAN.md §7](EXPERIMENT_PLAN.md).

Full table, per-item failures, and every verdict: [`results/RESULTS.md`](results/RESULTS.md).

## Run it

```bash
pip install -r requirements.txt
pytest -q                                   # 12 tests: baseline reproduces, all configs run offline
python scripts/run_experiments.py --phase 1 # E0-E1, no downloads
python scripts/run_experiments.py --phase 2 # E2-E3, downloads bge-small (~130 MB) once
python scripts/run_experiments.py           # everything whose requirements are met
python scripts/build_report.py              # -> results/RESULTS.md
```

E8 and E9 need a free [Groq API key](https://console.groq.com). Copy
`.env.example` to `.env` and add it. Every LLM response is cached to
`results/llm_cache.json`, so reruns are free and reproducible.

On Windows, `start.bat` sets up the environment and runs everything.

## Project structure

```
data/                     golden_dataset.json + policy_docs/ (unchanged from hr-policy-eval-harness)
configs/experiments.yaml  the experiment matrix: config + pre-registered hypothesis per experiment
src/rag_bench/
  chunking.py             section / contextual / fixed / sentence-window
  retrievers.py           TF-IDF, BM25, dense, hybrid RRF, cross-encoder rerank, HyDE
  guardrails.py           injection + ambiguity rules (controlled variable, unchanged)
  gates.py                decision layer + optional LLM scope gate
  calibration.py          per-retriever threshold search; fixed / in-sample / cross-validated modes
  metrics.py              pass/fail, Hit@k, MRR, OOS precision/recall, McNemar
  runner.py               runs one experiment -> results/runs/<id>.json
  llm.py                  cached Groq client
scripts/                  run_experiments.py, build_report.py
tests/                    reproduction, offline smoke tests with stub models, CV leakage check
results/                  per-run JSON, RESULTS.md, LLM cache
```

---
*Author: Robert Son · [GitHub](https://github.com/robertciceroson) · [LinkedIn](https://linkedin.com/in/robert-son-0b33b3bb)*
