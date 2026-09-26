# Results

Headline = **cross-validated accuracy** on the 73-item golden set (thresholds tuned on 4/5 of the data, scored on the held-out 1/5). In-sample = thresholds tuned on all 73 items (optimistic; comparable to the original 87.7%). One item = 1.4 points.

| ID | Experiment | CV acc | ±sd (10x CV) | In-sample | In-scope | OOS | Ambig. | Adv. | Doc Hit@1 | MRR | OOS precision | Latency p95 | vs E0 (p) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E0 | Baseline: TF-IDF, section chunks | **61/73 (83.6%)** | ±1.3% | 87.7% | 40/45 | 3/10 | 8/8 | 10/10 | 95.6% | 0.978 | 50.0% | 0.5 ms | - |
| E1 | BM25, section chunks | **57/73 (78.1%)** | ±1.6% | 83.6% | 38/45 | 3/10 | 6/8 | 10/10 | 95.6% | 0.978 | 33.3% | 0.1 ms | +3/-7 (p=0.34) |
| E2 | Dense (bge-small), section chunks | **65/73 (89.0%)** | ±1.3% | 90.4% | 43/45 | 8/10 | 4/8 | 10/10 | 97.8% | 0.989 | 61.5% | 17.7 ms | +10/-6 (p=0.45) |
| E3 | Hybrid RRF (BM25 + dense), section chunks | **64/73 (87.7%)** | ±1.4% | 91.8% | 43/45 | 5/10 | 6/8 | 10/10 | 97.8% | 0.989 | 62.5% | 10.9 ms | +7/-4 (p=0.55) |
| E4 | Dense, fixed 120-word chunks | **61/73 (83.6%)** | ±1.9% | 86.3% | 42/45 | 3/10 | 6/8 | 10/10 | 93.3% | 0.967 | 42.9% | 10.3 ms | +7/-7 (p=1.00) |
| E5 | Dense, sentence-window chunks | **65/73 (89.0%)** | ±1.3% | 90.4% | 43/45 | 8/10 | 4/8 | 10/10 | 97.8% | 0.989 | 61.5% | 9.8 ms | +10/-6 (p=0.45) |
| E6 | Dense, section + document-title headers | **60/73 (82.2%)** | ±1.6% | 87.7% | 42/45 | 3/10 | 5/8 | 10/10 | 95.6% | 0.978 | 37.5% | 9.1 ms | +8/-9 (p=1.00) |
| E7 | Hybrid + cross-encoder rerank | **69/73 (94.5%)** | ±0.5% | 95.9% | 42/45 | 9/10 | 8/8 | 10/10 | 100.0% | 1.000 | 75.0% | 1824.9 ms | +10/-2 (p=0.04) |
| E10 | Dense + cross-encoder rerank | **69/73 (94.5%)** | ±0.5% | 95.9% | 42/45 | 9/10 | 8/8 | 10/10 | 100.0% | 1.000 | 75.0% | 1310.6 ms | +10/-2 (p=0.04) |

**Baseline reproduction check:** E0 with the original hand-tuned parameters scores 64/73 (87.7%) (published: 64/73).

`vs E0` = items this experiment fixed / items it broke relative to the baseline, with an exact McNemar p-value. Decision rule (EXPERIMENT_PLAN.md): a real improvement needs >= 3 items gained, p < 0.10, adversarial still 10/10, and OOS precision no worse than the baseline. Anything else is reported as no measurable difference, however good the headline looks.

## E0: Baseline: TF-IDF, section chunks

*Hypothesis:* Reproduces the published 64/73 with the original hand-tuned params, and shows how much of 87.7% survives honest cross-validated calibration.

Chunks: 36 · index build 0.004 s · LLM calls this run: 0

| ID | Type | Question | Got | Top doc (conf) |
|---|---|---|---|---|
| HR-005 | in_scope | Do I need manager approval before calling in sick? | answer | expense_reimbursement_policy.md (0.3397) |
| HR-025 | in_scope | Who do I report a harassment concern to? | refuse_out_of_scope | code_of_conduct.md (0.1142) |
| HR-029 | in_scope | Am I allowed to tell a client what my coworkers get paid? | refuse_out_of_scope | code_of_conduct.md (0.1003) |
| HR-030 | in_scope | Does the confidentiality obligation end when I leave the company? | answer | parental_leave_policy.md (0.1628) |
| HR-044 | in_scope | How much does the company contribute to my HSA? | refuse_out_of_scope | benefits_enrollment_policy.md (0.1278) |
| HR-048 | out_of_scope | How do I apply for an internal job posting? | answer | parental_leave_policy.md (0.1368) |
| HR-049 | out_of_scope | Does the company offer tuition reimbursement? | answer | remote_work_policy.md (0.1519) |
| HR-050 | out_of_scope | What is the policy on bringing pets to the office? | answer | expense_reimbursement_policy.md (0.1602) |
| HR-051 | out_of_scope | How do I get a parking pass for the office? | answer | remote_work_policy.md (0.1498) |
| HR-052 | out_of_scope | What is the maternity leave policy for the UK office? | answer | parental_leave_policy.md (0.3283) |
| HR-054 | out_of_scope | How do I reset my company email password? | answer | remote_work_policy.md (0.2187) |
| HR-055 | out_of_scope | What's the policy on employee referral bonuses? | answer | parental_leave_policy.md (0.1011) |

**Verdict:** Reproduced exactly: 64/73 with the original hand-tuned thresholds. Under cross-validated calibration the honest baseline is 61/73 (83.6%, +/-1.3%), so about 3 points of the published 87.7% came from tuning the threshold on the same items it was scored on.

## E1: BM25, section chunks

*Hypothesis:* Better term weighting improves ranking slightly, but BM25 scores are unbounded and query-length dependent, so out-of-scope gating gets WORSE.

Chunks: 36 · index build 0.001 s · LLM calls this run: 0

| ID | Type | Question | Got | Top doc (conf) |
|---|---|---|---|---|
| HR-005 | in_scope | Do I need manager approval before calling in sick? | answer | expense_reimbursement_policy.md (4.1425) |
| HR-010 | in_scope | Am I eligible for a home office stipend, and how much is it? | clarify | remote_work_policy.md (8.8007) |
| HR-024 | in_scope | How long after approval does it take to get reimbursed? | clarify | pto_policy.md (4.3015) |
| HR-025 | in_scope | Who do I report a harassment concern to? | refuse_out_of_scope | code_of_conduct.md (3.4644) |
| HR-027 | in_scope | Do I need to disclose a side business I run on weekends? | refuse_out_of_scope | code_of_conduct.md (2.9267) |
| HR-029 | in_scope | Am I allowed to tell a client what my coworkers get paid? | refuse_out_of_scope | code_of_conduct.md (2.68) |
| HR-044 | in_scope | How much does the company contribute to my HSA? | refuse_out_of_scope | benefits_enrollment_policy.md (3.4311) |
| HR-046 | out_of_scope | What is the company's dress code policy? | answer | code_of_conduct.md (4.1507) |
| HR-048 | out_of_scope | How do I apply for an internal job posting? | answer | parental_leave_policy.md (3.3435) |
| HR-049 | out_of_scope | Does the company offer tuition reimbursement? | answer | remote_work_policy.md (3.4491) |
| HR-050 | out_of_scope | What is the policy on bringing pets to the office? | answer | expense_reimbursement_policy.md (4.5283) |
| HR-051 | out_of_scope | How do I get a parking pass for the office? | answer | expense_reimbursement_policy.md (3.7345) |
| HR-052 | out_of_scope | What is the maternity leave policy for the UK office? | answer | parental_leave_policy.md (5.8091) |
| HR-053 | out_of_scope | What is the company's severance pay policy? | answer | parental_leave_policy.md (2.7735) |
| HR-056 | ambiguous | How much PTO do I have? | refuse_out_of_scope | pto_policy.md (2.6064) |
| HR-059 | ambiguous | Am I eligible? | refuse_out_of_scope | remote_work_policy.md (2.1891) |

**Verdict:** Partly supported. Ranking was identical to TF-IDF (Hit@1 95.6%) and gating was worse (2 in-scope and 2 ambiguous items lost; OOS precision 33%), but +3/-7 vs E0 is not significant (p=0.34), so this is reported as no measurable difference.

## E2: Dense (bge-small), section chunks

*Hypothesis:* Embeddings fix both documented in-scope failures (synonym: 'calling in sick' ~ 'unplanned absence'; homonym: 'leave the company' vs 'parental leave') and raise OOS recall above 30%, because unrelated topics with shared words score lower semantically than lexically.

Chunks: 36 · index build 32.463 s · LLM calls this run: 0

| ID | Type | Question | Got | Top doc (conf) |
|---|---|---|---|---|
| HR-014 | in_scope | Can I access confidential company data from my personal phone? | answer | code_of_conduct.md (0.7262) |
| HR-027 | in_scope | Do I need to disclose a side business I run on weekends? | refuse_out_of_scope | code_of_conduct.md (0.6594) |
| HR-052 | out_of_scope | What is the maternity leave policy for the UK office? | answer | parental_leave_policy.md (0.7274) |
| HR-055 | out_of_scope | What's the policy on employee referral bonuses? | answer | code_of_conduct.md (0.688) |
| HR-057 | ambiguous | Can I take leave next month? | refuse_out_of_scope | pto_policy.md (0.6734) |
| HR-059 | ambiguous | Am I eligible? | refuse_out_of_scope | parental_leave_policy.md (0.6376) |
| HR-060 | ambiguous | How do I enroll? | refuse_out_of_scope | benefits_enrollment_policy.md (0.622) |
| HR-061 | ambiguous | What's the approval process? | refuse_out_of_scope | expense_reimbursement_policy.md (0.6811) |

**Verdict:** Mixed; does not pass the decision rule (+10/-6 vs E0, p=0.45). Fixed both documented retrieval misses ('calling in sick', 'leave the company') and raised OOS recall from 30% to 80%, but ambiguous handling fell from 8/8 to 4/8: short, vague questions ('Am I eligible?') score low on dense similarity and are refused as out-of- scope before the ambiguity check runs. It traded one failure type for another.

## E3: Hybrid RRF (BM25 + dense), section chunks

*Hypothesis:* Fusion keeps dense's semantic wins while recovering exact-term matches (dollar caps, day counts, form names). Ranking >= E2; gating ~= E2 because confidence still comes from the dense score.

Chunks: 36 · index build 1.818 s · LLM calls this run: 0

| ID | Type | Question | Got | Top doc (conf) |
|---|---|---|---|---|
| HR-005 | in_scope | Do I need manager approval before calling in sick? | answer | expense_reimbursement_policy.md (0.6829) |
| HR-025 | in_scope | Who do I report a harassment concern to? | refuse_out_of_scope | code_of_conduct.md (0.6355) |
| HR-046 | out_of_scope | What is the company's dress code policy? | answer | code_of_conduct.md (0.6545) |
| HR-049 | out_of_scope | Does the company offer tuition reimbursement? | answer | remote_work_policy.md (0.6457) |
| HR-052 | out_of_scope | What is the maternity leave policy for the UK office? | answer | parental_leave_policy.md (0.7189) |
| HR-053 | out_of_scope | What is the company's severance pay policy? | answer | parental_leave_policy.md (0.6517) |
| HR-055 | out_of_scope | What's the policy on employee referral bonuses? | answer | expense_reimbursement_policy.md (0.6513) |
| HR-060 | ambiguous | How do I enroll? | refuse_out_of_scope | benefits_enrollment_policy.md (0.622) |
| HR-061 | ambiguous | What's the approval process? | refuse_out_of_scope | expense_reimbursement_policy.md (0.6401) |

**Verdict:** Not supported. Ranking matched E2, but OOS recall fell to 50% and the 'calling in sick' lexical miss came back: fusing BM25 in reintroduces the keyword overlap that dense retrieval had filtered out. Not significant vs E0 (p=0.55).

## E4: Dense, fixed 120-word chunks

*Hypothesis:* Ignoring document structure hurts: chunks straddle sections and lose their headers, so Hit@1 drops versus section chunking.

Chunks: 19 · index build 2.349 s · LLM calls this run: 0

| ID | Type | Question | Got | Top doc (conf) |
|---|---|---|---|---|
| HR-014 | in_scope | Can I access confidential company data from my personal phone? | refuse_out_of_scope | code_of_conduct.md (0.5779) |
| HR-027 | in_scope | Do I need to disclose a side business I run on weekends? | refuse_out_of_scope | expense_reimbursement_policy.md (0.639) |
| HR-029 | in_scope | Am I allowed to tell a client what my coworkers get paid? | answer | expense_reimbursement_policy.md (0.6367) |
| HR-046 | out_of_scope | What is the company's dress code policy? | answer | code_of_conduct.md (0.6805) |
| HR-047 | out_of_scope | How do I request a transfer to a different department? | answer | pto_policy.md (0.6284) |
| HR-049 | out_of_scope | Does the company offer tuition reimbursement? | answer | expense_reimbursement_policy.md (0.6797) |
| HR-050 | out_of_scope | What is the policy on bringing pets to the office? | answer | remote_work_policy.md (0.607) |
| HR-052 | out_of_scope | What is the maternity leave policy for the UK office? | answer | parental_leave_policy.md (0.761) |
| HR-053 | out_of_scope | What is the company's severance pay policy? | answer | parental_leave_policy.md (0.6608) |
| HR-055 | out_of_scope | What's the policy on employee referral bonuses? | answer | expense_reimbursement_policy.md (0.687) |
| HR-059 | ambiguous | Am I eligible? | refuse_out_of_scope | benefits_enrollment_policy.md (0.6063) |
| HR-060 | ambiguous | How do I enroll? | refuse_out_of_scope | benefits_enrollment_policy.md (0.6136) |

**Verdict:** Supported. Fixed-size chunks ignore section boundaries (19 chunks instead of 36): Hit@1 fell to 93.3% and OOS recall back to 30%, returning to baseline level (61/73).

## E5: Dense, sentence-window chunks

*Hypothesis:* Smaller chunks match precise facts better (higher Hit@1 on multi-tier questions) but produce noisier confidence scores for gating.

Chunks: 36 · index build 1.562 s · LLM calls this run: 0

| ID | Type | Question | Got | Top doc (conf) |
|---|---|---|---|---|
| HR-014 | in_scope | Can I access confidential company data from my personal phone? | answer | code_of_conduct.md (0.7262) |
| HR-027 | in_scope | Do I need to disclose a side business I run on weekends? | refuse_out_of_scope | code_of_conduct.md (0.6594) |
| HR-052 | out_of_scope | What is the maternity leave policy for the UK office? | answer | parental_leave_policy.md (0.7274) |
| HR-055 | out_of_scope | What's the policy on employee referral bonuses? | answer | code_of_conduct.md (0.688) |
| HR-057 | ambiguous | Can I take leave next month? | refuse_out_of_scope | pto_policy.md (0.6734) |
| HR-059 | ambiguous | Am I eligible? | refuse_out_of_scope | parental_leave_policy.md (0.6376) |
| HR-060 | ambiguous | How do I enroll? | refuse_out_of_scope | benefits_enrollment_policy.md (0.622) |
| HR-061 | ambiguous | What's the approval process? | refuse_out_of_scope | expense_reimbursement_policy.md (0.6811) |

**Verdict:** Inconclusive: not tested as designed. The policy sections are short enough that 3-sentence windows produced the same 36 chunks as section chunking, so E5 is identical to E2. It needs a longer corpus (or a 1-sentence window) to be a real test.

## E6: Dense, section + document-title headers

*Hypothesis:* Prepending the policy name to each section is the cheapest possible fix for cross-document confusion; expect gains on the ambiguity/homonym items.

Chunks: 36 · index build 1.67 s · LLM calls this run: 0

| ID | Type | Question | Got | Top doc (conf) |
|---|---|---|---|---|
| HR-014 | in_scope | Can I access confidential company data from my personal phone? | refuse_out_of_scope | code_of_conduct.md (0.674) |
| HR-027 | in_scope | Do I need to disclose a side business I run on weekends? | refuse_out_of_scope | code_of_conduct.md (0.6384) |
| HR-034 | in_scope | My spouse and I both work here — do we each get the full 12 weeks? | answer | benefits_enrollment_policy.md (0.6815) |
| HR-046 | out_of_scope | What is the company's dress code policy? | answer | code_of_conduct.md (0.6653) |
| HR-047 | out_of_scope | How do I request a transfer to a different department? | answer | pto_policy.md (0.6317) |
| HR-049 | out_of_scope | Does the company offer tuition reimbursement? | answer | expense_reimbursement_policy.md (0.6802) |
| HR-050 | out_of_scope | What is the policy on bringing pets to the office? | answer | remote_work_policy.md (0.634) |
| HR-052 | out_of_scope | What is the maternity leave policy for the UK office? | answer | parental_leave_policy.md (0.7637) |
| HR-053 | out_of_scope | What is the company's severance pay policy? | answer | pto_policy.md (0.7081) |
| HR-055 | out_of_scope | What's the policy on employee referral bonuses? | answer | benefits_enrollment_policy.md (0.6878) |
| HR-057 | ambiguous | Can I take leave next month? | refuse_out_of_scope | parental_leave_policy.md (0.6763) |
| HR-059 | ambiguous | Am I eligible? | refuse_out_of_scope | benefits_enrollment_policy.md (0.6255) |
| HR-060 | ambiguous | How do I enroll? | refuse_out_of_scope | benefits_enrollment_policy.md (0.6355) |

**Verdict:** Not supported; it backfired (60/73, the worst dense variant). Prepending the policy title to every section made any question containing words like 'reimbursement' or 'leave' look like a strong match, so OOS recall fell to 30% and precision to 37.5%. The metadata reintroduced the lexical-overlap problem.

## E7: Hybrid + cross-encoder rerank

*Hypothesis:* A cross-encoder reads query and chunk together, so its score is a better relevance signal: best Hit@1 and the best non-LLM OOS gate.

Chunks: 36 · index build 36.196 s · LLM calls this run: 0

| ID | Type | Question | Got | Top doc (conf) |
|---|---|---|---|---|
| HR-027 | in_scope | Do I need to disclose a side business I run on weekends? | refuse_out_of_scope | code_of_conduct.md (-7.4312) |
| HR-029 | in_scope | Am I allowed to tell a client what my coworkers get paid? | refuse_out_of_scope | code_of_conduct.md (-8.2293) |
| HR-041 | in_scope | Can I change my benefits elections after having a baby? | refuse_out_of_scope | benefits_enrollment_policy.md (-3.0952) |
| HR-049 | out_of_scope | Does the company offer tuition reimbursement? | answer | remote_work_policy.md (-2.0828) |

**Verdict:** Supported, and passes the decision rule: 69/73 (94.5%, +/-0.5%), +10/-2 vs E0 (p=0.04), Hit@1 100%, OOS recall 90% at 75% precision, ambiguous 8/8, adversarial 10/10. Cost: about 1.8 s p95 latency on CPU, versus under 20 ms without reranking.

## E10: Dense + cross-encoder rerank

*Hypothesis:* Ablation added after Phase 4: hybrid alone lost to dense (E3 < E2), so is the reranker doing all the work in E7? If E10 ~= E7, BM25 adds nothing and the simpler dense + rerank pipeline should ship.

Chunks: 36 · index build 3.083 s · LLM calls this run: 0

| ID | Type | Question | Got | Top doc (conf) |
|---|---|---|---|---|
| HR-027 | in_scope | Do I need to disclose a side business I run on weekends? | refuse_out_of_scope | code_of_conduct.md (-7.4312) |
| HR-029 | in_scope | Am I allowed to tell a client what my coworkers get paid? | refuse_out_of_scope | code_of_conduct.md (-8.2293) |
| HR-041 | in_scope | Can I change my benefits elections after having a baby? | refuse_out_of_scope | benefits_enrollment_policy.md (-3.0952) |
| HR-049 | out_of_scope | Does the company offer tuition reimbursement? | answer | remote_work_policy.md (-2.0828) |

**Verdict:** Identical to E7 on every item, and faster (1.3 s vs 1.8 s p95): the cross-encoder does all the work and BM25 adds nothing. Caveat: the reranker re-scores the top 20 of only 36 chunks, so first-stage retrieval barely matters at this corpus size; on a large handbook it would. Recommended architecture so far. Remaining failures: 3 answerable questions refused (side business, sharing coworker pay, benefits change after a baby) and 1 out-of-scope question answered (tuition reimbursement).
