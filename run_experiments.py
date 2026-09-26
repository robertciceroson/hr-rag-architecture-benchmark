"""
Run the experiment matrix.

    python scripts/run_experiments.py              # everything whose requirements are met
    python scripts/run_experiments.py --only E0,E1
    python scripts/run_experiments.py --phase 2

Results land in results/runs/<id>.json; then run scripts/build_report.py.
"""
import argparse
import importlib.util
import os
import sys

import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "src"))

from rag_bench.llm import CachedLLM  # noqa: E402
from rag_bench.runner import needs_llm, run_experiment  # noqa: E402


def load_dotenv():
    path = os.path.join(ROOT, ".env")
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            if "=" in line and not line.strip().startswith("#"):
                k, v = line.strip().split("=", 1)
                os.environ.setdefault(k, v)


def missing_requirements(reqs):
    out = []
    for r in reqs:
        if r.isupper():
            if not os.environ.get(r):
                out.append(r)
        elif importlib.util.find_spec(r) is None:
            out.append(r)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated experiment ids")
    ap.add_argument("--phase", type=int)
    args = ap.parse_args()
    load_dotenv()

    with open(os.path.join(ROOT, "configs", "experiments.yaml"), encoding="utf-8") as f:
        exps = yaml.safe_load(f)["experiments"]
    if args.only:
        wanted = set(args.only.split(","))
        exps = [e for e in exps if e["id"] in wanted]
    if args.phase:
        exps = [e for e in exps if e["phase"] == args.phase]

    llm = None
    for exp in exps:
        missing = missing_requirements(exp.get("requires", []))
        if missing:
            print(f"{exp['id']:<4} SKIPPED - missing: {', '.join(missing)}")
            continue
        if needs_llm(exp) and llm is None:
            llm = CachedLLM(os.path.join(ROOT, "results", "llm_cache.json"))
        run_experiment(exp, llm=llm if needs_llm(exp) else None)


if __name__ == "__main__":
    main()
