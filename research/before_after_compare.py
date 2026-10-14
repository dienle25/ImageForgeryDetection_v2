"""Paired BEFORE (v1) / AFTER (v2) comparison on the SAME final-test images.

    python research/before_after_compare.py --v1 test_v1.jsonl --v1-config research/configs/config_v1_baseline.json \
        --v2 test_v2.jsonl --v2-config config.json --manifest M ... --splits test test_korus test_cocoglide \
        --out reports/test/before_after_compare.json

* only images scored without error by BOTH versions are used (v1 ran on a seeded subsample of generated/deepfake
  images, EXPERIMENT_LOG P09); identical decoded pixels counted once;
* verdicts re-derived from stored scores with each version's own thresholds (same rule as the apps);
* proportions with Wilson 95 % intervals; task A AUROC difference with a paired bootstrap over source groups.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ifd_core.metrics import auroc  # noqa: E402

ORDER = {"no_clear_sign": 1, "needs_review": 2, "suspicious": 3}


def source(i: str) -> str:
    if i.startswith("mei_"):
        return "ctrl_mei"
    if i.startswith("korus_"):
        return "korus"
    if i.startswith("cocoglide"):
        return "cocoglide"
    if i.startswith("openfake_"):
        return "openfake"
    if i.startswith("dff_"):
        return "dff_" + i.split("_")[1]
    return "other"


def wilson(k: int, n: int) -> list[float] | None:
    if n == 0:
        return None
    z = 1.959964
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return [max(0.0, c - h), min(1.0, c + h)]


def load_run(path: Path, cfg: dict) -> dict:
    th = {k: v["thresholds"] for k, v in cfg["detectors"].items() if v.get("enabled")}
    out = {}
    for l in path.read_text(encoding="utf-8").splitlines():
        if not l.strip():
            continue
        r = json.loads(l)
        if r.get("error"):
            continue
        dec = []
        for k, t in th.items():
            d = r["detectors"].get(k)
            if d is None or d["score"] is None:
                continue
            s = d["score"]
            dec.append(3 if s >= t["high"] else 2 if s >= t["low"] else 1)
        out[r["image_id"]] = {"overall": max(dec, default=1), "r": r}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v1", type=Path, required=True)
    ap.add_argument("--v1-config", type=Path, required=True)
    ap.add_argument("--v2", type=Path, required=True)
    ap.add_argument("--v2-config", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, action="append", required=True)
    ap.add_argument("--splits", nargs="+", required=True)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    meta = {}
    for mp in a.manifest:
        for l in mp.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                meta.setdefault(r["image_id"], r)
    c1 = json.loads(a.v1_config.read_text(encoding="utf-8"))
    c2 = json.loads(a.v2_config.read_text(encoding="utf-8"))
    r1, r2 = load_run(a.v1, c1), load_run(a.v2, c2)
    ids, seen_pix = [], set()
    for i in sorted(set(r1) & set(r2)):
        m = meta.get(i)
        if m is None or m.get("split") not in a.splits or m.get("exclude"):
            continue
        h = m.get("decoded_rgb_sha256")
        if h:
            if h in seen_pix:
                continue
            seen_pix.add(h)
        ids.append(i)
    out = {"v1_run": str(a.v1), "v2_run": str(a.v2), "n_paired_images": len(ids), "by_source": {}, "task_a": {}}
    by = defaultdict(list)
    for i in ids:
        by[source(i)].append(i)
    for src, lst in sorted(by.items()):
        blk = {}
        for lab, name in ((0, "real"), (1, "manipulated_or_generated")):
            sel = [i for i in lst if int(meta[i]["label"]) == lab]
            if not sel:
                continue
            n = len(sel)
            row = {"n": n}
            for ver, rr in (("v1", r1), ("v2", r2)):
                o = np.array([rr[i]["overall"] for i in sel])
                k3, k2, k1 = int((o >= 3).sum()), int((o >= 2).sum()), int((o == 1).sum())
                row[ver] = {"suspicious": k3 / n, "suspicious_ci95": wilson(k3, n),
                            "review_or_more": k2 / n, "review_or_more_ci95": wilson(k2, n),
                            "no_clear_sign": k1 / n}
            s1 = np.array([r1[i]["overall"] >= 3 for i in sel])
            s2 = np.array([r2[i]["overall"] >= 3 for i in sel])
            row["paired_suspicious"] = {"v1_only": int((s1 & ~s2).sum()), "v2_only": int((~s1 & s2).sum()),
                                        "both": int((s1 & s2).sum()), "neither": int((~s1 & ~s2).sum())}
            blk[name] = row
        out["by_source"][src] = blk
    # task A: AUROC of v1 PSCC vs v2 Mesorch on the same images, paired bootstrap over groups
    rng = np.random.default_rng(0)
    for src in ("ctrl_mei", "korus", "cocoglide"):
        lst = by.get(src, [])
        if not lst:
            continue
        y = np.array([int(meta[i]["label"]) for i in lst])
        s1 = np.array([r1[i]["r"]["detectors"]["pscc_net"]["score"] for i in lst], float)
        s2 = np.array([r2[i]["r"]["detectors"]["mesorch"]["score"] for i in lst], float)
        g = np.array([meta[i].get("group_id", i) for i in lst])
        uniq = np.unique(g)
        idx = {u: np.nonzero(g == u)[0] for u in uniq}
        diffs = []
        for _ in range(a.boot):
            pick = rng.choice(uniq, size=len(uniq), replace=True)
            ii = np.concatenate([idx[u] for u in pick])
            if y[ii].min() == y[ii].max():
                continue
            diffs.append(auroc(y[ii], s2[ii]) - auroc(y[ii], s1[ii]))
        blk = {"n_pos": int(y.sum()), "n_neg": int((y == 0).sum()), "auroc_v1_pscc": auroc(y, s1),
               "auroc_v2_mesorch": auroc(y, s2), "difference_v2_minus_v1": auroc(y, s2) - auroc(y, s1),
               "difference_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))] if diffs else None}
        f1v1, f1v2, fa1, fa2 = [], [], [], []
        for i in lst:
            p1 = r1[i]["r"]["detectors"]["pscc_net"].get("pixel") or {}
            p2 = r2[i]["r"]["detectors"]["mesorch"].get("pixel") or {}
            if "error" in p1 or "error" in p2 or not p1 or not p2:
                continue
            if int(meta[i]["label"]) == 1 and meta[i].get("mask"):
                f1v1.append(p1.get("f1") or 0.0)
                f1v2.append(p2.get("f1") or 0.0)
            elif int(meta[i]["label"]) == 0:
                fa1.append(p1["frac_pred"])
                fa2.append(p2["frac_pred"])
        if f1v1:
            blk["pixel_macro_f1"] = {"v1": float(np.mean(f1v1)), "v2": float(np.mean(f1v2)), "n": len(f1v1)}
        if fa1:
            blk["false_activation_on_authentic"] = {"v1_mean_fraction": float(np.mean(fa1)), "v2_mean_fraction": float(np.mean(fa2)),
                                                    "n": len(fa1)}
        out["task_a"][src] = blk
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print("paired images", len(ids))


if __name__ == "__main__":
    main()
