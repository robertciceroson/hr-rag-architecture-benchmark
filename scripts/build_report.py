"""
Build results/RESULTS.md from every results/runs/*.json.

Verdicts live in configs/experiments.yaml (`verdict:` per experiment), NOT in
RESULTS.md, because this script regenerates RESULTS.md from scratch every run.

Headline metric = cross-validated accuracy (see calibration.py for why).
Each experiment also gets an exact McNemar test against E0 on the same
73 items, so small differences aren't over-claimed.
"""
import glob
import json
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))
from rag_bench.metrics import mcnemar_exact  # noqa: E402


def pct(x):
    return "-" if x is None else f"{x:.1%}"


def main():
    runs = {}
    for path in sorted(glob.glob(os.path.join(ROOT, "results", "runs", "*.json"))):
        r = json.load(open(path, encoding="utf-8"))
        runs[r["id"]] = r
    if not runs:
        sys.exit("No runs found. Run scripts/run_experiments.py first.")
    order = sorted(runs, key=lambda k: int(k[1:]))
    with open(os.path.join(ROOT, "configs", "experiments.yaml"), encoding="utf-8") as f:
        verdicts = {e["id"]: e.get("verdict", "").strip()
                    for e in yaml.safe_load(f)["experiments"]}
    base = runs.get("E0")

    L = ["# Results\n",
         "Headline = **cross-validated accuracy** on the 73-item golden set "
         "(thresholds tuned on 4/5 of the data, scored on the held-out 1/5). "
         "In-sample = thresholds tuned on all 73 items (optimistic; comparable "
         "to the original 87.7%). One item = 1.4 points.\n",
         "| ID | Experiment | CV acc | ±sd (10x CV) | In-sample | In-scope | OOS | Ambig. | Adv. "
         "| Doc Hit@1 | MRR | OOS precision | Latency p95 | vs E0 (p) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for k in order:
        r = runs[k]
        cv, bc = r["cv"], r["cv"]["by_category"]
        cat = lambda c: f"{bc[c]['passed']}/{bc[c]['total']}"  # noqa: E731
        sig = "-"
        if base and k != "E0":
            ids = list(cv["per_item"])
            t = mcnemar_exact([base["cv"]["per_item"][i]["passed"] for i in ids],
                              [cv["per_item"][i]["passed"] for i in ids])
            sig = f"+{t['experiment_only']}/-{t['baseline_only']} (p={t['p_value']:.2f})"
        L.append(
            f"| {k} | {r['name']} | **{cv['passed']}/{cv['total']} ({pct(cv['accuracy'])})** "
            f"| ±{r['cv_repeated']['std']:.1%} | {pct(r['in_sample']['accuracy'])} "
            f"| {cat('in_scope')} | {cat('out_of_scope')} | {cat('ambiguous')} "
            f"| {cat('adversarial')} | {pct(r['retrieval']['doc_hit@1'])} "
            f"| {r['retrieval']['doc_mrr']:.3f} | {pct(cv['oos_detection']['precision'])} "
            f"| {r['retrieval']['latency_ms_p95']:.1f} ms | {sig} |")

    if base and "fixed" in base:
        fx = base["fixed"]
        L += ["", f"**Baseline reproduction check:** E0 with the original hand-tuned "
              f"parameters scores {fx['passed']}/{fx['total']} ({pct(fx['accuracy'])}) "
              f"(published: 64/73)."]

    L += ["", "`vs E0` = items this experiment fixed / items it broke relative to the "
          "baseline, with an exact McNemar p-value. Decision rule (EXPERIMENT_PLAN.md): a real "
          "improvement needs >= 3 items gained, p < 0.10, adversarial still 10/10, and OOS "
          "precision no worse than the baseline. Anything else is reported as no measurable "
          "difference, however good the headline looks.", ""]

    for k in order:
        r = runs[k]
        L += [f"## {k}: {r['name']}", "", f"*Hypothesis:* {r['hypothesis'].strip()}", "",
              f"Chunks: {r['n_chunks']} · index build {r['index_build_s']} s · "
              f"LLM calls this run: {r['llm_calls']}", ""]
        if r["cv_failures"]:
            L += ["| ID | Type | Question | Got | Top doc (conf) |", "|---|---|---|---|---|"]
            for f in r["cv_failures"]:
                q = f["question"].replace("|", "\\|")
                L.append(f"| {f['id']} | {f['case_type']} | {q} | {f['action']} "
                         f"| {f['top_doc']} ({f['top_conf']}) |")
        v = verdicts.get(k) or "_Pending: add a `verdict:` to this experiment in configs/experiments.yaml._"
        L += ["", f"**Verdict:** {v}", ""]

    out = os.path.join(ROOT, "results", "RESULTS.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L))
    print(f"Wrote {out}")


if __name__ == "__main__":
    main()
