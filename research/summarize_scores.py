"""Summarize image-level scores from a results JSONL against one or more manifests.

Usage:
  python research/summarize_scores.py RESULTS.jsonl --manifest M1.jsonl [--manifest M2.jsonl]
         [--split dev] [--score score | --score details.max] [--by generator] [--neg-by real_source]
         [--json-out summary.json]

Positives of each `--by` value are compared against ALL negatives of the same split, so the
per-subgroup AUROC answers: "how well does this score separate edits of kind X from all reals".
CI = percentile bootstrap over group_id (source lineage), 1000 resamples, seed 0.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ifd_core.metrics import auroc, average_precision, bootstrap_ci, confusion, threshold_for_fpr  # noqa: E402


def get(rec: dict, key: str):
    cur = rec
    for part in key.split("."):
        if not isinstance(cur, dict) or part not in cur:
            return None
        cur = cur[part]
    return cur


def load(results: Path, manifests: list[Path], split: str | None, score_key: str, none_as: float | None = None):
    meta = {}
    for m in manifests:
        for line in m.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                meta[r["image_id"]] = r
    rows, missing, errors, dups = [], 0, 0, 0
    seen, seen_pix = set(), {}
    for line in results.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r["image_id"] in seen:
            continue
        seen.add(r["image_id"])
        if r.get("error"):
            errors += 1
            continue
        m = meta.get(r["image_id"])
        if m is None:
            missing += 1
            continue
        if split and m.get("split") != split:
            continue
        if m.get("exclude"):
            load.excluded = getattr(load, "excluded", 0) + 1
            continue
        # identical decoded pixels are evaluated once (e.g. the authentic twin of two edit pairs)
        h = m.get("decoded_rgb_sha256")
        if h:
            if h in seen_pix:
                if seen_pix[h] != int(m["label"]):
                    raise SystemExit(f"label conflict for identical pixels: {r['image_id']}")
                dups += 1
                continue
            seen_pix[h] = int(m["label"])
        s = get(r, score_key)
        if s is None and none_as is not None:
            s = none_as  # e.g. "no face found" -> detector does not flag
            load.none_replaced = getattr(load, "none_replaced", 0) + 1
        if s is None:
            errors += 1
            continue
        rows.append({**m, "_score": float(s), "_res": r})
    load.duplicates_skipped = dups
    return rows, missing, errors


def block(y, s, g, with_ci=True):
    y = np.asarray(y)
    s = np.asarray(s, dtype=np.float64)
    out = {"n": int(len(y)), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum())}
    out["auroc"] = auroc(y, s)
    out["ap"] = average_precision(y, s)
    if with_ci and out["auroc"] is not None:
        out["auroc_ci95"] = bootstrap_ci(auroc, y, s, g)
    if out["n_neg"] and out["n_pos"]:
        for f in (0.01, 0.05, 0.10, 0.20):
            t = threshold_for_fpr(y, s, f)
            c = confusion(y, s, t)
            out[f"tpr_at_fpr{int(f*100):02d}"] = {"threshold": t, "tpr": c["recall"], "fpr": c["fpr"]}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("results", type=Path)
    ap.add_argument("--manifest", type=Path, action="append", required=True)
    ap.add_argument("--split", default=None)
    ap.add_argument("--score", default="score")
    ap.add_argument("--by", default=None, help="manifest field to break positives down by")
    ap.add_argument("--neg-by", default=None, help="manifest field to break negatives down by")
    ap.add_argument("--json-out", type=Path, default=None)
    ap.add_argument("--none-as", type=float, default=None, help="score to use when the result has score=None")
    a = ap.parse_args()

    rows, missing, errors = load(a.results, a.manifest, a.split, a.score, a.none_as)
    if not rows:
        raise SystemExit("no rows")
    y = np.array([int(r["label"]) for r in rows])
    s = np.array([r["_score"] for r in rows])
    g = np.array([r.get("group_id", r["image_id"]) for r in rows])
    summary = {"results": str(a.results), "split": a.split, "score_key": a.score,
               "missing_from_manifest": missing, "errors_or_missing_score": errors,
               "identical_pixel_duplicates_skipped": getattr(load, "duplicates_skipped", 0),
               "none_scores_replaced": getattr(load, "none_replaced", 0), "none_as": a.none_as,
               "manifest_excluded_rows": getattr(load, "excluded", 0),
               "overall": block(y, s, g)}
    secs = [r["_res"].get("seconds") for r in rows if r["_res"].get("seconds") is not None]
    if secs:
        summary["seconds_p50_p95"] = [float(np.percentile(secs, 50)), float(np.percentile(secs, 95))]
    neg = y == 0
    if a.by:
        per = {}
        vals = defaultdict(list)
        for i, r in enumerate(rows):
            if y[i] == 1:
                vals[str(r.get(a.by))].append(i)
        for v, idx in sorted(vals.items()):
            sel = np.zeros(len(rows), bool)
            sel[idx] = True
            sel |= neg
            b = block(y[sel], s[sel], g[sel], with_ci=len(idx) >= 20)
            b["pos_score_median"] = float(np.median(s[idx]))
            per[v] = b
        summary["by_" + a.by] = per
    if a.neg_by:
        per = {}
        vals = defaultdict(list)
        for i, r in enumerate(rows):
            if y[i] == 0:
                vals[str(r.get(a.neg_by))].append(i)
        t05 = summary["overall"].get("tpr_at_fpr05", {}).get("threshold")
        for v, idx in sorted(vals.items()):
            ss = s[idx]
            per[v] = {"n": len(idx), "median": float(np.median(ss)), "p90": float(np.percentile(ss, 90)),
                      "frac_above_overall_fpr05_threshold": (float(np.mean(ss >= t05)) if t05 is not None else None)}
        summary["neg_by_" + a.neg_by] = per
    txt = json.dumps(summary, indent=1, ensure_ascii=False)
    if a.json_out:
        a.json_out.parent.mkdir(parents=True, exist_ok=True)
        a.json_out.write_text(txt + "\n", encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
