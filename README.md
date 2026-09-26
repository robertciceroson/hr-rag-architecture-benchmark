# Experiment Plan: Which RAG Architecture Actually Works for HR Policy Q&A?

**Owner:** Robert Son · **Status:** Phase 1 complete, Phases 2–5 ready to run
**Companion repos:** [HR-Policy-QA-Bot](https://github.com/robertciceroson/HR-Policy-QA-Bot) (the product) ·
[hr-policy-eval-harness](https://github.com/robertciceroson/hr-policy-eval-harness) (the golden dataset and baseline)

---

## 1. Why this project exists

The eval harness scored the baseline RAG pipeline at 87.7% and named its two
weaknesses precisely:

1. **Out-of-scope detection: 30% (3/10).** "Tuition reimbursement", "parking
   pass", and "UK maternity leave" share vocabulary with real policies, so no
   confidence threshold separates them from answerable questions.
2. **Two in-scope retrieval misses**: a synonym problem ("calling in sick" vs.
   "unplanned absence") and a homonym problem ("leave the company" vs.
   "parental leave").

The harness recommended a fix but never tested it. This project tests that
recommendation and the main alternatives from the *AI Engineering Guidebook*
(Daily Dose of DS, 2025) under controlled conditions, so the answer to "which
RAG design should we ship?" is measured, not assumed.

## 2. Questions this project answers

| # | Question | Experiments |
|---|---|---|
| Q1 | Does semantic retrieval fix the synonym/homonym misses and improve scope detection? | E0 → E2 |
| Q2 | Is hybrid (lexical + semantic) retrieval worth the extra moving part? | E2 → E3 |
| Q3 | How much does chunking strategy matter on structured policy documents? | E2 → E4, E5, E6 |
| Q4 | Does a cross-encoder reranker earn its latency? | E3 → E7 |
| Q5 | Does HyDE help, or does it make hallucination-prone questions worse? | E2 → E8 |
| Q6 | Does an LLM scope gate solve the 30% out-of-scope problem without refusing real questions? | E3 → E9 |

## 3. Experimental design

### Held constant (controlled variables)
- **Golden dataset:** the same 73 hand-curated items (45 in-scope, 10 out-of-scope, 8 ambiguous, 10 adversarial). Copied unchanged into `data/`.
- **Knowledge base:** the same 6 synthetic policy documents.
- **Guardrail rules:** the injection and ambiguity-pattern rules are copied unchanged (`src/rag_bench/guardrails.py`). Adversarial items are blocked before retrieval, so they must stay at **10/10 in every experiment**. Any drop is a bug, and the test suite enforces it.
- **Decision order:** injection → scope gate → ambiguity → answer, the same as the original pipeline.
- **Scoring rules:** in-scope passes only if the system answers *and* the top document is correct. Every other type passes on the correct action. The rules are identical to `evaluate.py`.

### Changed, one at a time
Retriever, chunking strategy, reranking, query transformation (HyDE), or an
LLM scope gate. Each experiment names the experiment it's compared against
(`compare_to` in `configs/experiments.yaml`).

### The fairness problem, and how it's handled
Each retriever scores on a different scale. TF-IDF cosine runs about 0–0.6.
BM25 is unbounded. bge cosine runs about 0.5–0.9, and cross-encoder scores
are logits. A threshold tuned for one is meaningless for the others, so every
experiment gets its **own** gate parameters (scope threshold, ambiguity
margin, ambiguity minimum confidence), found by grid search.

Tuning on the same items you report on inflates the score, so there are three reporting modes:

| Mode | How thresholds are set | Used for |
|---|---|---|
| `fixed` | Original hand-tuned values (0.10 / 0.02 / 0.20) | E0 only: proves the baseline reproduces |
| `in_sample` | Tuned on all 73 items | Optimistic ceiling; comparable to the published 87.7% |
| **`cv`** | **Stratified 5-fold: tune on 4 folds, score the 5th** | **Headline number for every comparison** |

Retrieval runs once per question. The decision layer then replays on the
cached features, so calibration re-runs no retrieval and makes no new LLM
calls.

### Metrics
| Metric | What it tells you |
|---|---|
| **CV accuracy** (+ by case type) | Headline: end-to-end correctness under honest calibration |
| ±sd over 10 repeated CV splits | How stable the headline is |
| Doc Hit@1, Hit@3, MRR (in-scope) | Pure retrieval quality, independent of gating |
| OOS precision / recall | Scope detection as a classifier. Precision matters: a false refusal of a real question is a product failure too |
| Latency p95, LLM calls, index build time | Cost side of the tradeoff |
| Exact McNemar test vs. E0 | Whether a gain is larger than chance on the same 73 items |

### Decision rule (set before seeing results)
With 73 items, one item is 1.4 points. An experiment counts as a **real
improvement** only if:
1. CV accuracy beats its comparison point by **≥ 3 items**, **and**
2. McNemar p < 0.10 vs. E0 (lenient, given the small n; report the exact value), **and**
3. adversarial stays 10/10 and OOS precision does not drop below the baseline's.

Anything else gets reported as "no measurable difference," even when the
headline number is higher.

## 4. Experiment matrix and pre-registered hypotheses

Hypotheses are written down **before** running. A wrong hypothesis is still a
finding: record it honestly in `results/RESULTS.md`.

| ID | Change | Guidebook ref | Hypothesis |
|---|---|---|---|
| **E0** | TF-IDF, section chunks (control) | — | Reproduces 64/73 fixed; CV is lower |
| **E1** | BM25 | p.113 | Ranking ≈ same; gating worse (unbounded scores) |
| **E2** | Dense bge-small (same model as production bot) | p.107 | Fixes both in-scope misses; OOS recall > 30% |
| **E3** | Hybrid RRF (BM25 + dense) | p.126 | Ranking ≥ E2; gating ≈ E2 |
| **E4** | Dense, fixed 120-word chunks | p.118 | Hit@1 drops: chunks lose section headers |
| **E5** | Dense, sentence-window chunks | p.118 | Better on precise facts, noisier gating |
| **E6** | Dense, doc title prepended to each section | p.146 | Cheapest fix for cross-document confusion |
| **E7** | Hybrid + cross-encoder rerank | p.126 | Best Hit@1 and best non-LLM scope gate |
| **E8** | HyDE (Llama 3.3 70B) + dense | p.131 | Helps vague questions, **hurts** OOS: the LLM invents a plausible "dress code policy" that then matches real text |
| **E9** | Hybrid + LLM scope gate | p.128 | Lifts OOS recall to 80%+ with no false refusals: tests the harness's own recommendation |

**Stretch (not scaffolded):** agentic RAG with query rewriting and retry (p.128),
and generation-quality scoring with an LLM-backed generator plus G-Eval
(p.332). Add these only after E0–E9 are written up.

## 5. Results so far (Phase 1, run 2026-09-26)

| ID | CV accuracy | In-sample | OOS recall | Doc Hit@1 |
|---|---|---|---|---|
| E0 fixed params | 64/73 (87.7%) | — | 30% | 95.6% |
| **E0 cv** | **61/73 (83.6%) ± 1.3%** | 87.7% | 30% | 95.6% |
| E1 BM25 | 57/73 (78.1%) ± 1.6% | 83.6% | 30% | 95.6% |

**Finding 1:** the published 87.7% was partly a product of tuning the
threshold on the test set. The honest baseline is **83.6%**, and that is
the number every later experiment has to beat.
**Finding 2:** BM25 behaved as hypothesized. Ranking was identical
(Hit@1 95.6%), but its unbounded scores made the gates worse (−4 items:
2 in-scope and 2 ambiguous items lost).

## 6. Run plan

| Phase | Experiments | Where | Time | Cost |
|---|---|---|---|---|
| 0. Setup | clone, `pip install -r requirements.txt`, `pytest -q` | your PC | 30 min | $0 |
| 1. Lexical | E0, E1 | anywhere | **done** | $0 |
| 2. Semantic | E2, E3 | your PC (downloads ~130 MB model once) | 1 hr | $0 |
| 3. Chunking | E4, E5, E6 | your PC | 1 hr | $0 |
| 4. Reranking | E7 | your PC (~90 MB model) | 30 min | $0 |
| 5. LLM-assisted | E8, E9 | your PC + `GROQ_API_KEY` | 1 hr | $0 on Groq free tier (~150 calls, cached) |
| 6. Write-up | verdicts in RESULTS.md, README findings, LinkedIn post | — | 2–3 hrs | $0 |

About one weekend in total. After each phase: `python scripts/build_report.py`,
fill in each experiment's *Verdict* line, then commit the results and
`results/llm_cache.json`.

**Phase 3 note:** E4–E6 use dense retrieval. If E3 (hybrid) clearly wins
Phase 2, change their `retriever` block to the hybrid spec so the chunking
comparison runs on the best retriever.

## 7. Threats to validity (put these in the README; it shows judgment)

- **Small n.** 73 items, only 10 out-of-scope. A single item moves OOS recall by 10 points. Hence the McNemar tests and the ≥3-item rule.
- **Ambiguity patterns are dataset-aware.** The 8 regexes in `guardrails.py` were written alongside the ambiguous test items, so 8/8 ambiguous is partly in-sample by construction. It stays constant across experiments, so comparisons remain fair, but don't claim it as a general ambiguity-detection result.
- **Synthetic corpus.** Six short, clean markdown policies. Real handbooks (long PDFs, tables, amendments) would stress chunking far more, so E4–E6 likely *understate* chunking effects.
- **Document-level scoring.** A pass means the right *document* ranked first. It doesn't prove the right *section* was found or that an LLM would phrase the answer correctly (see the stretch goal).
- **LLM nondeterminism.** Temperature is 0 and every completion is cached and committed, so E8 and E9 reproduce exactly from the cache.

## 8. Deliverables

1. `results/RESULTS.md`: full comparison table, per-experiment failures, verdicts
2. README "Findings" section: 3–5 bullets, each tied to a number
3. A **recommendation** for the production HR-Policy-QA-Bot: which architecture to ship, and at what latency and cost
4. LinkedIn post (use the saved voice): the "87.7% → 83.6% honest baseline → X% best architecture" story
5. Resume bullet (after results), e.g. *"Benchmarked 10 RAG architectures (lexical, dense, hybrid, reranking, HyDE, LLM scope gating) against a 73-case golden dataset with cross-validated calibration and significance testing; identified [winner], lifting out-of-scope detection from 30% to X%."*
