"""Summarise robustness runs (research/eval_robustness.py) against the untransformed final-test scores.

    python research/summarize_robustness.py --robust R.jsonl [--robust R2.jsonl] --base test_v2.jsonl \
        --config config.json --out reports/test/robustness_summary.json

For every (detector, transform): the SAME images are compared with their untransformed score from the product run
(--base), using the locked thresholds of --config.  Reports AUROC when both labels are present, the fraction at or
above `high` ("Nghi vấn") and `low` ("Cần kiểm tra thêm" or above) per label, and paired flips across `high`.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ifd_core.metrics import auroc, bootstrap_ci  # noqa: E402


def frac(x):
    return float(np.mean(x)) if len(x) else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--robust", type=Path, action="append", required=True)
    ap.add_argument("--base", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    cfg = json.loads(a.config.read_text(encoding="utf-8"))
    th = {k: v["thresholds"] for k, v in cfg["detectors"].items() if v.get("enabled")}
    base = {}
    for l in a.base.read_text(encoding="utf-8").splitlines():
        if l.strip():
            r = json.loads(l)
            if not r.get("error"):
                base[r["image_id"]] = r
    groups = defaultdict(list)
    errors = defaultdict(int)
    for rp in a.robust:
        for l in rp.read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            if r.get("error"):
                errors[(r["detector"], r["transform"])] += 1
                continue
            groups[(r["detector"], r["transform"])].append(r)
    out = {"config": str(a.config), "base_run": str(a.base), "thresholds": th, "results": {}}
    for (det, tr), rs in sorted(groups.items()):
        t = th[det]
        rows = []
        for r in rs:
            b = base.get(r["image_id"])
            bd = (b or {}).get("detectors", {}).get(det)
            if bd is None:
                continue
            s_new = -1.0 if r["score"] is None else float(r["score"])      # None = no face found -> never flagged
            s_old = -1.0 if bd["score"] is None else float(bd["score"])
            rows.append((int(r["label"]), s_old, s_new, r["image_id"]))
        if not rows:
            continue
        y = np.array([x[0] for x in rows])
        so = np.array([x[1] for x in rows])
        sn = np.array([x[2] for x in rows])
        blk = {"n": int(len(rows)), "n_pos": int(y.sum()), "n_neg": int((y == 0).sum()),
               "errors": errors.get((det, tr), 0)}
        if y.any() and (y == 0).any():
            ids = [x[3] for x in rows]
            blk["auroc_original"] = auroc(y, so)
            blk["auroc_transformed"] = auroc(y, sn)
            blk["auroc_transformed_ci95"] = bootstrap_ci(auroc, y, sn, ids)
        for lab, name in ((0, "real_or_authentic"), (1, "manipulated_or_generated")):
            m = y == lab
            if not m.any():
                continue
            blk[name] = {
                "n": int(m.sum()),
                "ge_high_original": frac(so[m] >= t["high"]), "ge_high_transformed": frac(sn[m] >= t["high"]),
                "ge_low_original": frac(so[m] >= t["low"]), "ge_low_transformed": frac(sn[m] >= t["low"]),
                "crossed_up_to_suspicious": int(((so[m] < t["high"]) & (sn[m] >= t["high"])).sum()),
                "crossed_down_from_suspicious": int(((so[m] >= t["high"]) & (sn[m] < t["high"])).sum()),
                "median_score_original": float(np.median(so[m])), "median_score_transformed": float(np.median(sn[m])),
            }
            if det == "face_clip":
                blk[name]["no_face_original"] = frac(so[m] < 0)
                blk[name]["no_face_transformed"] = frac(sn[m] < 0)
        out["results"].setdefault(det, {})[tr] = blk
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for det, d in out["results"].items():
        for tr, b in d.items():
            print(det, tr, b["n"], {k: (round(v, 4) if isinstance(v, float) else v) for k, v in b.items()
                                    if k in ("auroc_original", "auroc_transformed")})


if __name__ == "__main__":
    main()
