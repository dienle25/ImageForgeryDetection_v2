"""Derive per-detector operating thresholds from PRODUCT scores on the development split only.

Rule (QUALITY_TARGETS.yaml, operating_point_policy):
  high = smallest threshold whose FPR is <= 5 %  on EVERY development negative source of that detector's task
  low  = smallest threshold whose FPR is <= 20 % on EVERY development negative source
(using every source separately keeps one easy source from hiding a hard one).

  python research/derive_thresholds.py --dev RUN.jsonl [--dev RUN2.jsonl] --manifest M.jsonl ... --out thresholds.json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ifd_core.metrics import auroc, confusion  # noqa: E402

# which development sources supply negatives / positives for each detector (by manifest "dataset"-like key)
TASK_OF = {"mesorch": "local_manipulation", "commfor_aigen": "ai_generated", "face_clip": "face_deepfake",
           "pscc_net": "local_manipulation"}


def source_of(m: dict) -> str:
    iid = m["image_id"]
    if iid.startswith("mei_"):
        return "ctrl_mei"
    if iid.startswith("korus_"):
        return "korus"
    if iid.startswith("openfake_"):
        return "openfake:" + ("real_" + str(m.get("real_source")) if int(m["label"]) == 0 else "fake")
    if iid.startswith("dff_"):
        return "dff:" + iid.split("_")[1]
    if iid.startswith("cocoglide"):
        return "cocoglide"
    return "other"


def relevant(det: str, m: dict) -> str | None:
    """'neg' / 'pos' / None for this detector's own task on the dev data."""
    src = source_of(m)
    lab = int(m["label"])
    task = TASK_OF[det]
    if lab == 0:
        if task == "local_manipulation":
            # every detector runs on every uploaded image, so every real photo of the dev split is a negative
            # (camera photos, ImageNet/DOCCI photos, real portraits) - not only the authentic twins of edits
            return "neg"
        if task == "ai_generated":
            return "neg"          # every real photo is a negative for "fully AI-generated"
        if task == "face_deepfake":
            return "neg"          # real faces (no-face images score None -> not flagged)
    else:
        if task == "local_manipulation":
            return "pos" if src in ("ctrl_mei", "korus", "dff:inpainting") else None
        if task == "ai_generated":
            return "pos" if src in ("openfake:fake", "dff:text2img") else None
        if task == "face_deepfake":
            return "pos" if src in ("dff:insight", "dff:inpainting") else None
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", type=Path, action="append", required=True)
    ap.add_argument("--manifest", type=Path, action="append", required=True)
    ap.add_argument("--high-fpr", type=float, default=0.05)
    ap.add_argument("--low-fpr", type=float, default=0.20)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    meta = {}
    for mp in a.manifest:
        for l in mp.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                meta[r["image_id"]] = r
    recs = {}
    for dp in a.dev:
        for l in dp.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                recs[r["image_id"]] = r
    out = {"rule": f"high: FPR<={a.high_fpr} on every dev negative source; low: FPR<={a.low_fpr} on every source",
           "dev_runs": [str(p) for p in a.dev], "detectors": {}}
    dets = sorted({k for r in recs.values() for k in r.get("detectors", {})})
    for det in dets:
        negs, poss = defaultdict(list), defaultdict(list)
        errors = 0
        seen_pix = set()
        for iid, r in recs.items():
            m = meta.get(iid)
            if m is None or m.get("exclude") or "split" not in m or not str(m["split"]).startswith("dev"):
                continue
            h = m.get("decoded_rgb_sha256")
            if h:
                if h in seen_pix:
                    continue
                seen_pix.add(h)
            if r.get("error"):
                errors += 1
                continue
            role = relevant(det, m)
            if role is None:
                continue
            d = r["detectors"].get(det)
            if d is None:
                continue
            s = d["score"]
            s = -1.0 if s is None else float(s)   # not applicable (no face) => never flagged
            (negs if role == "neg" else poss)[source_of(m)].append(s)
        if not negs:
            continue
        hi = float(np.clip(max(_t(v, a.high_fpr) for v in negs.values()), 0.0, 1.0))
        lo = float(np.clip(max(_t(v, a.low_fpr) for v in negs.values()), 0.0, 1.0))
        lo = min(lo, hi)
        info = {"task": TASK_OF[det], "high": hi, "low": lo, "errors": errors,
                "neg_sources": {k: {"n": len(v), "fpr_at_high": float(np.mean(np.array(v) >= hi)),
                                    "fpr_at_low": float(np.mean(np.array(v) >= lo)),
                                    "uncertain_band_fraction": float(np.mean((np.array(v) >= lo) & (np.array(v) < hi)))}
                                for k, v in sorted(negs.items())},
                "pos_sources": {k: {"n": len(v), "tpr_at_high": float(np.mean(np.array(v) >= hi)),
                                    "tpr_at_low": float(np.mean(np.array(v) >= lo)),
                                    "uncertain_band_fraction": float(np.mean((np.array(v) >= lo) & (np.array(v) < hi)))}
                                for k, v in sorted(poss.items())}}
        allneg = [x for v in negs.values() for x in v]
        allpos = [x for v in poss.values() for x in v]
        if allpos:
            y = [0] * len(allneg) + [1] * len(allpos)
            s = allneg + allpos
            info["dev_auroc_pooled"] = auroc(y, s)
            info["dev_confusion_at_high"] = confusion(y, s, hi)
        out["detectors"][det] = info
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=1, ensure_ascii=False))


def _t(neg_scores, fpr):
    """smallest threshold with FPR <= fpr on these negatives (scores >= t are flagged)."""
    v = np.sort(np.asarray(neg_scores, dtype=np.float64))
    k = int(np.floor(fpr * v.size))           # allowed false positives
    if k >= v.size:
        return float(v[0])
    t = v[v.size - k - 1]
    return float(np.nextafter(t, np.inf))


if __name__ == "__main__":
    main()
