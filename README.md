# HR RAG Architecture Benchmark

*Which RAG design actually works for HR policy Q&A? Measured, not assumed.*

This is the third project in a series:

| Repo | Role |
|---|---|
| [HR-Policy-QA-Bot](https://github.com/robertciceroson/HR-Policy-QA-Bot) | **Build**: LangChain + FAISS + FastEmbed + Groq/Llama 3.3 70B RAG chatbot |
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
| E9 | Hybrid + LLM scope gate |

## How it's kept fair

- **One variable per experiment.** The dataset, documents, guardrail rules, and scoring rules never change. Adversarial items must stay 10/10 in every run, and a test enforces this.
- **Per-retriever calibration.** Each retriever gets its own scope and ambiguity thresholds, because TF-IDF, BM25, embedding, and cross-encoder scores live on different scales.
- **Cross-validated headline.** Thresholds are tuned on 4/5 of the items and scored on the held-out 1/5. Tuning on the test set is what inflated the original 87.7%.
- **Significance testing.** An exact McNemar test compares each experiment with the baseline on the same 73 items. With n = 73, a gain of one or two items is noise and gets reported as noise.

## Findings so far

| | Result |
|---|---|
| Baseline, original hand-tuned thresholds | 64/73 (87.7%): reproduces the published score |
| **Baseline, honest cross-validated calibration** | **61/73 (83.6%) ± 1.3%**: the real number to beat |
| BM25 | 57/73 (78.1%): same ranking, worse gating (hypothesis supported) |
| E2–E9 | *Pending: see [EXPERIMENT_PLAN.md §6](EXPERIMENT_PLAN.md)* |

Full table and per-item failures: [`results/RESULTS.md`](results/RESULTS.md).

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
