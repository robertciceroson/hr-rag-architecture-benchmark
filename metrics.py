"""Scoring: the original pass/fail rules, plus retrieval and scope metrics."""
from collections import defaultdict
from typing import Dict, List

import numpy as np

from .gates import ItemFeatures, is_correct

CASE_TYPES = ("in_scope", "out_of_scope", "ambiguous", "adversarial")


def score_actions(feats: List[ItemFeatures], actions: List[str]) -> Dict:
    per_item = [is_correct(f, a) for f, a in zip(feats, actions)]
    tot, ok = defaultdict(int), defaultdict(int)
    for f, p in zip(feats, per_item):
        tot[f.case_type] += 1
        ok[f.case_type] += int(p)

    # Scope detection as a binary classifier: positive = "refuse_out_of_scope"
    is_oos = [f.case_type == "out_of_scope" for f in feats]
    said_oos = [a == "refuse_out_of_scope" for a in actions]
    tp = sum(t and s for t, s in zip(is_oos, said_oos))
    fp = sum((not t) and s for t, s in zip(is_oos, said_oos))
    fn = sum(t and (not s) for t, s in zip(is_oos, said_oos))
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0

    return {
        "passed": sum(per_item),
        "total": len(per_item),
        "accuracy": sum(per_item) / len(per_item),
        "by_category": {c: {"passed": ok[c], "total": tot[c],
                            "accuracy": ok[c] / tot[c] if tot[c] else None}
                        for c in CASE_TYPES},
        "oos_detection": {"precision": prec, "recall": rec,
                          "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
                          "false_refusals_of_answerable": fp},
        "per_item": {f.item_id: {"passed": p, "action": a}
                     for f, p, a in zip(feats, per_item, actions)},
    }


def retrieval_quality(feats: List[ItemFeatures], dataset: List[dict]) -> Dict:
    """Pure retrieval metrics on in_scope items, independent of gating:
    did the right DOCUMENT come back, and how high?"""
    expected = {d["id"]: d["expected_source_doc"] for d in dataset}
    hit1, hit3, rr = [], [], []
    for f in feats:
        if f.case_type != "in_scope":
            continue
        docs = f.ranked_docs or []
        exp = expected[f.item_id]
        rank = docs.index(exp) + 1 if exp in docs else None
        hit1.append(rank == 1)
        hit3.append(rank is not None and rank <= 3)
        rr.append(1.0 / rank if rank else 0.0)
    lat = [f.latency_ms for f in feats]
    return {
        "doc_hit@1": float(np.mean(hit1)),
        "doc_hit@3": float(np.mean(hit3)),
        "doc_mrr": float(np.mean(rr)),
        "latency_ms_mean": float(np.mean(lat)),
        "latency_ms_p95": float(np.percentile(lat, 95)),
    }


def mcnemar_exact(a_pass: List[bool], b_pass: List[bool]) -> Dict:
    """Paired test on the SAME 73 items: is B's improvement over A bigger than
    chance? Only items where exactly one system passed carry information."""
    from scipy.stats import binomtest
    b = sum(x and not y for x, y in zip(a_pass, b_pass))   # A right, B wrong
    c = sum(y and not x for x, y in zip(a_pass, b_pass))   # B right, A wrong
    p = binomtest(min(b, c), b + c, 0.5).pvalue if b + c else 1.0
    return {"baseline_only": b, "experiment_only": c, "p_value": float(p)}
