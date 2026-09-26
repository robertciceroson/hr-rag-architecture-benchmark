# Results

Headline = **cross-validated accuracy** on the 73-item golden set (thresholds tuned on 4/5 of the data, scored on the held-out 1/5). In-sample = thresholds tuned on all 73 items (optimistic; comparable to the original 87.7%). One item = 1.4 points.

| ID | Experiment | CV acc | ±sd (10x CV) | In-sample | In-scope | OOS | Ambig. | Adv. | Doc Hit@1 | MRR | OOS precision | Latency p95 | vs E0 (p) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| E0 | Baseline: TF-IDF, section chunks | **61/73 (83.6%)** | ±1.3% | 87.7% | 40/45 | 3/10 | 8/8 | 10/10 | 95.6% | 0.978 | 50.0% | 0.5 ms | - |
| E1 | BM25, section chunks | **57/73 (78.1%)** | ±1.6% | 83.6% | 38/45 | 3/10 | 6/8 | 10/10 | 95.6% | 0.978 | 33.3% | 0.1 ms | +3/-7 (p=0.34) |

**Baseline reproduction check:** E0 with the original hand-tuned parameters scores 64/73 (87.7%) (published: 64/73).

`vs E0` = items this experiment fixed / items it broke relative to the baseline, with an exact McNemar p-value. With 73 items, treat p > 0.05 as "not distinguishable from the baseline", however good the headline looks.

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

*Verdict:* _TODO - supported / not supported, and why (1-3 sentences)._

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

*Verdict:* _TODO - supported / not supported, and why (1-3 sentences)._
