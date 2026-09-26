"""Runs one experiment end to end and writes results/runs/<id>.json."""
import json
import os
import time
from typing import Dict, Optional

from .calibration import apply, cross_validate, repeated_cv_accuracy, search_params
from .chunking import build_chunks
from .gates import extract_features, llm_scope_check
from .guardrails import is_injection
from .metrics import retrieval_quality, score_actions
from .retrievers import build_retriever, timed_search

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")


def load_dataset():
    with open(os.path.join(DATA, "golden_dataset.json"), encoding="utf-8") as f:
        return json.load(f)


def needs_llm(spec: dict) -> bool:
    return spec.get("llm_scope_gate", False) or "hyde" in json.dumps(spec["retriever"])


def run_experiment(exp: Dict, llm=None, embedder=None, encoder=None,
                   write: bool = True, verbose: bool = True) -> Dict:
    dataset = load_dataset()
    t0 = time.perf_counter()
    chunks = build_chunks(os.path.join(DATA, "policy_docs"), exp["chunking"]["strategy"],
                          **exp["chunking"].get("params", {}))
    retriever = build_retriever(exp["retriever"], chunks, llm=llm,
                                embedder=embedder, encoder=encoder)
    index_s = time.perf_counter() - t0

    feats = []
    for item in dataset:
        hits, ms = timed_search(retriever, item["question"], top_k=5)
        llm_oos = False
        if exp.get("llm_scope_gate") and not is_injection(item["question"]):
            llm_oos = llm_scope_check(llm, item["question"], hits)
        feats.append(extract_features(item, hits, ms, llm_oos))

    result = {
        "id": exp["id"],
        "name": exp["name"],
        "hypothesis": exp.get("hypothesis", ""),
        "config": {k: exp[k] for k in ("chunking", "retriever") if k in exp} |
                  {"llm_scope_gate": exp.get("llm_scope_gate", False)},
        "n_chunks": len(chunks),
        "index_build_s": round(index_s, 3),
        "llm_calls": getattr(llm, "calls", 0) if llm else 0,
        "retrieval": retrieval_quality(feats, dataset),
    }

    cv_actions, fold_params = cross_validate(feats)
    result["cv"] = score_actions(feats, cv_actions) | {"fold_params": fold_params}
    result["cv_repeated"] = repeated_cv_accuracy(feats)

    in_params = search_params(feats)
    result["in_sample"] = score_actions(feats, apply(feats, in_params)) | {"params": in_params}

    if exp.get("fixed_params"):
        result["fixed"] = score_actions(feats, apply(feats, exp["fixed_params"])) | \
                          {"params": exp["fixed_params"]}

    # Human-readable failure list (cv mode) for the write-up
    by_id = {d["id"]: d for d in dataset}
    result["cv_failures"] = [
        {"id": f.item_id, "case_type": f.case_type, "question": by_id[f.item_id]["question"],
         "action": a, "expected_doc": by_id[f.item_id]["expected_source_doc"],
         "top_doc": f.top_doc, "top_conf": round(f.top_conf, 4)}
        for f, a in zip(feats, cv_actions) if not result["cv"]["per_item"][f.item_id]["passed"]
    ]
    if hasattr(retriever, "hypotheticals"):
        result["hyde_samples"] = dict(list(retriever.hypotheticals.items())[:5])

    if write:
        os.makedirs(os.path.join(RESULTS, "runs"), exist_ok=True)
        with open(os.path.join(RESULTS, "runs", f"{exp['id']}.json"), "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)
    if verbose:
        cv, ins = result["cv"], result["in_sample"]
        line = (f"{exp['id']:<4} {exp['name']:<42} CV {cv['passed']}/{cv['total']} "
                f"({cv['accuracy']:.1%})  in-sample {ins['accuracy']:.1%}  "
                f"OOS recall {cv['oos_detection']['recall']:.0%}  "
                f"Hit@1 {result['retrieval']['doc_hit@1']:.1%}")
        if "fixed" in result:
            line += f"  fixed {result['fixed']['passed']}/{result['fixed']['total']}"
        print(line)
    return result
