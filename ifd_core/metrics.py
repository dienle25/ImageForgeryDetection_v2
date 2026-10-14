"""Dependency-light (NumPy) evaluation metrics with explicit definitions.

AUROC : Mann-Whitney U statistic with average ranks for ties (equals area under ROC).
AP    : sum_n (R_n - R_{n-1}) P_n over descending score thresholds (same as sklearn average_precision_score).
CI    : percentile bootstrap resampling *groups* (source lineage) with replacement.
"""
from __future__ import annotations

import numpy as np


def _ranks(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x, kind="mergesort")
    sx = x[order]
    _, start, counts = np.unique(sx, return_index=True, return_counts=True)
    avg = start + (counts + 1) / 2.0
    r = np.empty(len(x), dtype=np.float64)
    r[order] = np.repeat(avg, counts)
    return r


def auroc(labels, scores) -> float | None:
    y = np.asarray(labels).astype(bool)
    s = np.asarray(scores, dtype=np.float64)
    npos, nneg = int(y.sum()), int((~y).sum())
    if npos == 0 or nneg == 0:
        return None
    r = _ranks(s)
    return float((r[y].sum() - npos * (npos + 1) / 2) / (npos * nneg))


def average_precision(labels, scores) -> float | None:
    y = np.asarray(labels).astype(bool)
    s = np.asarray(scores, dtype=np.float64)
    if y.sum() == 0:
        return None
    order = np.argsort(-s, kind="mergesort")
    ys, ss = y[order], s[order]
    # group ties: evaluate only at distinct thresholds
    distinct = np.r_[np.nonzero(np.diff(ss))[0], len(ss) - 1]
    tp = np.cumsum(ys)[distinct]
    fp = (distinct + 1) - tp
    precision = tp / (tp + fp)
    recall = tp / y.sum()
    prev_r = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - prev_r) * precision))


def confusion(labels, scores, threshold: float) -> dict:
    y = np.asarray(labels).astype(bool)
    p = np.asarray(scores, dtype=np.float64) >= threshold
    tp, tn = int((y & p).sum()), int((~y & ~p).sum())
    fp, fn = int((~y & p).sum()), int((y & ~p).sum())
    div = lambda a, b: (a / b) if b else None
    return {"threshold": threshold, "tp": tp, "tn": tn, "fp": fp, "fn": fn,
            "precision": div(tp, tp + fp), "recall": div(tp, tp + fn), "f1": div(2 * tp, 2 * tp + fp + fn),
            "fpr": div(fp, fp + tn), "fnr": div(fn, fn + tp), "accuracy": div(tp + tn, len(y)),
            "balanced_accuracy": (None if (tp + fn) == 0 or (tn + fp) == 0 else 0.5 * (tp / (tp + fn) + tn / (tn + fp)))}


def threshold_for_fpr(labels, scores, target_fpr: float) -> float:
    """Smallest threshold whose FPR on these (validation) negatives is <= target."""
    y = np.asarray(labels).astype(bool)
    neg = np.sort(np.asarray(scores, dtype=np.float64)[~y])
    if neg.size == 0:
        raise ValueError("no negatives")
    k = int(np.floor(target_fpr * neg.size))  # allowed false positives
    if k >= neg.size:
        return float(neg[0])
    # threshold strictly above the (k+1)-th largest negative
    t = neg[neg.size - k - 1]
    return float(np.nextafter(t, np.inf))


def best_f1_threshold(labels, scores) -> tuple[float, float]:
    s = np.asarray(scores, dtype=np.float64)
    cands = np.unique(s)
    best = (0.5, -1.0)
    for t in cands:
        f1 = confusion(labels, s, float(t))["f1"] or 0.0
        if f1 > best[1]:
            best = (float(t), f1)
    return best


def bootstrap_ci(fn, labels, scores, groups, n: int = 1000, seed: int = 0, alpha: float = 0.05):
    y = np.asarray(labels)
    s = np.asarray(scores, dtype=np.float64)
    g = np.asarray(groups)
    uniq = np.unique(g)
    idx_by_g = {u: np.nonzero(g == u)[0] for u in uniq}
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(n):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([idx_by_g[u] for u in pick])
        v = fn(y[idx], s[idx])
        if v is not None:
            vals.append(v)
    if not vals:
        return None
    lo, hi = np.percentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return [float(lo), float(hi)]


def brier(labels, probs) -> float:
    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(probs, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def ece(labels, probs, bins: int = 10) -> float:
    y = np.asarray(labels, dtype=np.float64)
    p = np.asarray(probs, dtype=np.float64)
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (p >= lo) & (p < hi) if hi < 1 else (p >= lo) & (p <= hi)
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)
