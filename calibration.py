"""
Fair threshold calibration.

Why this exists: every retriever produces scores on a different scale
(TF-IDF cosine ~0-0.6, BM25 unbounded, bge cosine ~0.5-0.9, cross-encoder
logits). A threshold hand-tuned for TF-IDF is meaningless for the others, so
each experiment gets its gate parameters searched on its own scores.

But tuning on the same 73 items you report on inflates the score. The
original baseline's 0.10 threshold was tuned that way, so 87.7% is an
in-sample number. This module reports three modes:

  fixed      - use given params (only meaningful for the TF-IDF baseline)
  in_sample  - tune on all 73 items (optimistic ceiling; comparable to 87.7%)
  cv         - stratified 5-fold: tune on 4 folds, score the held-out fold.
               THIS is the headline number used to compare experiments.
"""
from typing import Dict, List, Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .gates import ItemFeatures, decide, is_correct

MAX_CANDIDATES = 40


def _candidates(values: Sequence[float], extra: Sequence[float]) -> np.ndarray:
    v = np.unique([x for x in values if np.isfinite(x)])
    if len(v) > 1:
        v = np.concatenate([v, (v[:-1] + v[1:]) / 2])  # include midpoints
        v = np.unique(v)
    if len(v) > MAX_CANDIDATES:
        v = np.quantile(v, np.linspace(0, 1, MAX_CANDIDATES))
    return np.unique(np.concatenate([v, np.array(extra, dtype=float)]))


def search_params(feats: List[ItemFeatures]) -> Dict[str, float]:
    """Grid-search (oos_threshold, amb_margin, amb_min_conf) to maximize
    accuracy on `feats`. Ties go to the middle of the tied set, which is less
    sensitive to noise than taking the edge of the range."""
    live = [f for f in feats if not f.injection]
    top = np.array([f.top_conf for f in live])
    gap = np.array([f.gap_to_next_doc for f in live])

    thr_c = _candidates(top, [-np.inf])
    mar_c = _candidates(gap, [-np.inf])        # -inf disables the margin rule
    min_c = _candidates(top, [np.inf])         # +inf disables the margin rule

    ct = np.array([f.case_type for f in live])
    doc_ok = np.array([f.doc_correct for f in live])
    llm_oos = np.array([f.llm_out_of_scope for f in live])
    pat = np.array([f.pattern_ambiguous for f in live])

    best, best_sets = -1, []
    for thr in thr_c:
        oos = (top < thr) | llm_oos
        # clarify[m, c, i] for items that survive the scope gate
        margin_hit = (gap[None, :] <= mar_c[:, None])          # (M, I)
        conf_hit = (top[None, :] >= min_c[:, None])            # (C, I)
        clar = (~oos)[None, None, :] & (pat[None, None, :] |
                                         (margin_hit[:, None, :] & conf_hit[None, :, :]))
        answer = (~oos)[None, None, :] & ~clar
        correct = np.where(ct == "in_scope", answer & doc_ok,
                  np.where(ct == "out_of_scope", np.broadcast_to(oos, answer.shape),
                  np.where(ct == "ambiguous", clar, False)))
        scores = correct.sum(axis=-1)
        m = scores.max()
        if m > best:
            best, best_sets = m, []
        if m == best:
            for mi, ci in zip(*np.where(scores == m)):
                best_sets.append((thr, mar_c[mi], min_c[ci]))
    thr, mar, mn = best_sets[len(best_sets) // 2]
    return {"oos_threshold": float(thr), "amb_margin": float(mar), "amb_min_conf": float(mn)}


def apply(feats: List[ItemFeatures], params: Dict[str, float]) -> List[str]:
    return [decide(f, **params) for f in feats]


def cross_validate(feats: List[ItemFeatures], n_splits: int = 5, seed: int = 42):
    """Returns held-out actions for every item plus the params chosen per fold."""
    y = [f.case_type for f in feats]
    actions: List[str] = [""] * len(feats)
    fold_params = []
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for train_idx, test_idx in skf.split(np.zeros(len(y)), y):
        params = search_params([feats[i] for i in train_idx])
        fold_params.append(params)
        for i in test_idx:
            actions[i] = decide(feats[i], **params)
    return actions, fold_params


def repeated_cv_accuracy(feats: List[ItemFeatures], repeats: int = 10) -> Dict[str, float]:
    accs = []
    for r in range(repeats):
        actions, _ = cross_validate(feats, seed=r)
        accs.append(np.mean([is_correct(f, a) for f, a in zip(feats, actions)]))
    return {"mean": float(np.mean(accs)), "std": float(np.std(accs)), "repeats": repeats}
