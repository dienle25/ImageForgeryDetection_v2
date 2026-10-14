"""Metrics for the product run (dev or final test): per task, per dataset, per group; product-level verdicts.

  python research/final_report.py --run RUN.jsonl [--run RUN2.jsonl] --manifest M.jsonl ... \
      --config config.json --splits test test_korus test_cocoglide --out metrics.json

Definitions (docs/DATA_PROTOCOL.md §8): AUROC (ties averaged) with 95 % CI from 1000 bootstrap resamples of
source groups (seed 0); confusion at the config's `high` ("Nghi vấn") and `low` ("Cần kiểm tra thêm" or above);
abstention band = [low, high); pixel metrics at the config's pixel threshold on images with a mask; false
activation on authentic images. Images whose decoded pixels are identical are counted once.
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ifd_core.metrics import auroc, average_precision, bootstrap_ci, confusion  # noqa: E402

ORDER = {"no_clear_sign": 1, "needs_review": 2, "suspicious": 3}


def source(m: dict) -> str:
    i = m["image_id"]
    if i.startswith("mei_"):
        return "ctrl_mei"
    if i.startswith("korus_"):
        return "korus"
    if i.startswith("cocoglide"):
        return "cocoglide"
    if i.startswith("openfake_"):
        return "openfake"
    if i.startswith("dff_"):
        return "dff_" + i.split("_")[1]          # dff_wiki / dff_insight / dff_inpainting / dff_text2img
    return "other"


def load(runs, manifests, splits):
    meta = {}
    for mp in manifests:
        for l in Path(mp).read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                meta.setdefault(r["image_id"], r)
    recs, errors, seen_pix, dups, excluded = {}, [], set(), 0, 0
    for rp in runs:
        for l in Path(rp).read_text(encoding="utf-8").splitlines():
            if not l.strip():
                continue
            r = json.loads(l)
            m = meta.get(r["image_id"])
            if m is None or (splits and m.get("split") not in splits):
                continue
            if m.get("exclude"):
                excluded += 1
                continue
            if r.get("error"):
                errors.append({"image_id": r["image_id"], "error": r["error"][:200]})
                continue
            h = m.get("decoded_rgb_sha256")
            if h:
                if h in seen_pix:
                    dups += 1
                    continue
                seen_pix.add(h)
            recs[r["image_id"]] = {"m": m, "r": r, "src": source(m)}
    return recs, errors, dups, excluded


def score_of(rec, det):
    d = rec["r"]["detectors"].get(det)
    if d is None:
        return None
    return -1.0 if d["score"] is None else float(d["score"])    # None = not applicable (no face) => never flagged


def binary_block(pos, neg, th, groups_pos=None, groups_neg=None, ci=True):
    """pos/neg: lists of scores. Returns AUROC/AP/CI + confusion at high & low + abstention."""
    y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
    s = np.r_[pos, neg].astype(float)
    out = {"n_pos": len(pos), "n_neg": len(neg)}
    if len(pos) and len(neg):
        out["auroc"] = auroc(y, s)
        out["ap"] = average_precision(y, s)
        if ci:
            g = np.r_[groups_pos if groups_pos is not None else [f"p{i}" for i in range(len(pos))],
                      groups_neg if groups_neg is not None else [f"n{i}" for i in range(len(neg))]]
            out["auroc_ci95"] = bootstrap_ci(auroc, y, s, g)
    for name, t in (("at_high", th["high"]), ("at_low", th["low"])):
        out[name] = confusion(y, s, t)
    band = (s >= th["low"]) & (s < th["high"])
    covered = ~band
    out["abstain_fraction"] = float(band.mean()) if len(s) else None
    out["coverage"] = float(covered.mean()) if len(s) else None
    if covered.any():
        pred = s[covered] >= th["high"]
        out["accuracy_on_covered"] = float((pred == (y[covered] == 1)).mean())
        out["fpr_on_covered_neg"] = float(pred[y[covered] == 0].mean()) if (y[covered] == 0).any() else None
        out["tpr_on_covered_pos"] = float(pred[y[covered] == 1].mean()) if (y[covered] == 1).any() else None
    return out


def neg_only_block(neg, th):
    s = np.asarray(neg, float)
    return {"n": int(s.size), "fpr_at_high": float((s >= th["high"]).mean()) if s.size else None,
            "flag_or_review_at_low": float((s >= th["low"]).mean()) if s.size else None,
            "fp_at_high": int((s >= th["high"]).sum())}


def pixel_block(recs, det, pixel_t):
    f1, iou, auc, tp, fp, fn = [], [], [], 0, 0, 0
    act, act_imgs = [], []
    for rec in recs:
        d = rec["r"]["detectors"].get(det, {})
        px = d.get("pixel")
        if not px or "error" in px:
            continue
        if int(rec["m"]["label"]) == 1 and rec["m"].get("mask"):
            f1.append(px["f1"] or 0.0)
            iou.append(px["iou"] or 0.0)
            if px.get("auc") is not None:
                auc.append(px["auc"])
            tp, fp, fn = tp + px["tp"], fp + px["fp"], fn + px["fn"]
        elif int(rec["m"]["label"]) == 0:
            act.append(px["frac_pred"])
    out = {"pixel_threshold": pixel_t, "n_masked_positives": len(f1)}
    if f1:
        out.update(macro_f1=float(np.mean(f1)), macro_iou=float(np.mean(iou)),
                   micro_f1=float(2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else None,
                   mean_pixel_auc=float(np.mean(auc)) if auc else None,
                   median_f1=float(np.median(f1)), fraction_f1_ge_0_5=float(np.mean(np.array(f1) >= 0.5)))
    if act:
        a = np.asarray(act)
        out.update(n_authentic=int(a.size), authentic_mean_false_activation=float(a.mean()),
                   authentic_fraction_images_activation_gt_1pct=float((a > 0.01).mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", type=Path, action="append", required=True)
    ap.add_argument("--manifest", type=Path, action="append", required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--splits", nargs="+", default=None)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--recompute-decisions", action="store_true",
                    help="re-derive per-detector and overall decisions from the stored scores with THIS config's "
                         "thresholds (to report an alternative operating point fixed before the test run)")
    a = ap.parse_args()
    cfg = json.loads(a.config.read_text(encoding="utf-8"))
    th = {k: v["thresholds"] for k, v in cfg["detectors"].items() if v.get("enabled")}
    recs, errors, dups, excluded = load(a.run, a.manifest, set(a.splits) if a.splits else None)
    if a.recompute_decisions:
        for rec in recs.values():
            worst = 0
            for k, d in rec["r"]["detectors"].items():
                if k not in th:
                    continue
                if d["score"] is None:
                    d["decision"] = "not_applicable"
                    continue
                s_ = float(d["score"])
                d["decision"] = "suspicious" if s_ >= th[k]["high"] else ("needs_review" if s_ >= th[k]["low"] else "no_clear_sign")
                worst = max(worst, ORDER[d["decision"]])
            rec["r"]["overall_decision"] = {3: "suspicious", 2: "needs_review", 1: "no_clear_sign", 0: "no_clear_sign"}[worst]
    by_src = defaultdict(list)
    for rec in recs.values():
        by_src[rec["src"]].append(rec)
    out = {"runs": [str(p) for p in a.run], "splits": a.splits, "config": str(a.config),
           "decisions_recomputed_from_scores": bool(a.recompute_decisions),
           "thresholds": th, "n_images": len(recs), "errors": errors, "identical_pixel_duplicates_skipped": dups,
           "excluded_by_manifest": excluded, "counts_by_source": {k: len(v) for k, v in sorted(by_src.items())},
           "tasks": {}, "product_level": {}}

    def grp(rec):
        return rec["m"].get("group_id", rec["m"]["image_id"])

    def split_pn(rs, det, is_pos):
        pos = [r for r in rs if is_pos(r)]
        neg = [r for r in rs if not is_pos(r)]
        return pos, neg

    # ---------------- Task A: local manipulation (mesorch in v2; pscc_net for the v1 baseline) ----------------
    det = next((k for k in ("mesorch", "pscc_net") if k in th), None)
    if det is not None:
        A = {}
        for src in ("ctrl_mei", "korus", "cocoglide"):
            rs = by_src.get(src, [])
            if not rs:
                continue
            pos, neg = split_pn(rs, det, lambda r: int(r["m"]["label"]) == 1)
            blk = binary_block([score_of(r, det) for r in pos], [score_of(r, det) for r in neg], th[det],
                               [grp(r) for r in pos], [grp(r) for r in neg])
            blk["pixel"] = pixel_block(rs, det, th[det].get("pixel", 0.5))
            if src == "ctrl_mei":
                for field in ("operation", "size_bucket", "post_processing"):
                    sub = {}
                    vals = sorted({str(r["m"].get(field)) for r in pos})
                    for v in vals:
                        pv = [r for r in pos if str(r["m"].get(field)) == v]
                        nv = neg if field != "post_processing" else [r for r in neg if str(r["m"].get(field)) == v]
                        sub[v] = binary_block([score_of(r, det) for r in pv], [score_of(r, det) for r in nv], th[det],
                                              [grp(r) for r in pv], [grp(r) for r in nv], ci=len(pv) >= 20)
                    blk["by_" + field] = sub
            if src == "korus":
                sub = {}
                for cam in sorted({r["m"].get("camera") for r in rs}):
                    pv = [r for r in pos if r["m"].get("camera") == cam]
                    nv = [r for r in neg if r["m"].get("camera") == cam]
                    sub[cam] = binary_block([score_of(r, det) for r in pv], [score_of(r, det) for r in nv], th[det], ci=False)
                blk["by_camera"] = sub
            A[src] = blk
        out["tasks"]["local_manipulation"] = A
        out["local_manipulation_detector"] = det

    # ---------------- Task B: fully AI-generated (commfor) ----------------
    det = "commfor_aigen"
    if det in th:
        B = {}
        of = by_src.get("openfake", [])
        if of:
            pos, neg = split_pn(of, det, lambda r: int(r["m"]["label"]) == 1)
            blk = binary_block([score_of(r, det) for r in pos], [score_of(r, det) for r in neg], th[det])
            blk["by_generator"] = {}
            for gname in sorted({r["m"].get("generator") for r in pos}):
                pv = [r for r in pos if r["m"].get("generator") == gname]
                b = binary_block([score_of(r, det) for r in pv], [score_of(r, det) for r in neg], th[det], ci=len(pv) >= 20)
                blk["by_generator"][gname] = {k: b[k] for k in ("n_pos", "auroc", "auroc_ci95", "at_high", "at_low") if k in b}
            blk["by_real_source"] = {s: neg_only_block([score_of(r, det) for r in neg if r["m"].get("real_source") == s], th[det])
                                     for s in sorted({str(r["m"].get("real_source")) for r in neg})}
            B["openfake_holdout_generators"] = blk
        t2i = by_src.get("dff_text2img", [])
        wiki = by_src.get("dff_wiki", [])
        if t2i and wiki:
            B["dff_text2img_vs_wiki"] = binary_block([score_of(r, det) for r in t2i], [score_of(r, det) for r in wiki], th[det],
                                                     [grp(r) for r in t2i], [grp(r) for r in wiki])
        cam = {}
        for src in ("ctrl_mei", "korus", "cocoglide", "openfake", "dff_wiki"):
            neg = [r for r in by_src.get(src, []) if int(r["m"]["label"]) == 0]
            if neg:
                cam[src] = neg_only_block([score_of(r, det) for r in neg], th[det])
        B["false_positive_on_real_photos_by_source"] = cam
        cross = {}
        for src in ("ctrl_mei", "korus", "cocoglide", "dff_insight", "dff_inpainting"):
            pos = [r for r in by_src.get(src, []) if int(r["m"]["label"]) == 1]
            if pos:
                s = np.array([score_of(r, det) for r in pos])
                cross[src] = {"n": int(s.size), "fraction_ge_high": float((s >= th[det]["high"]).mean())}
        B["activation_on_partially_edited_images_(not_errors)"] = cross
        out["tasks"]["ai_generated"] = B

    # ---------------- Task C: face deepfake (face_clip) ----------------
    det = "face_clip"
    if det in th:
        C = {}
        wiki = by_src.get("dff_wiki", [])
        for fake_src, name in (("dff_insight", "faceswap_insightface_vs_real"), ("dff_inpainting", "face_inpainting_sd_vs_real"),
                               ("dff_text2img", "fully_generated_faces_vs_real_(cross_task)")):
            fk = by_src.get(fake_src, [])
            if fk and wiki:
                C[name] = binary_block([score_of(r, det) for r in fk], [score_of(r, det) for r in wiki], th[det],
                                       [grp(r) for r in fk], [grp(r) for r in wiki])
        if wiki:
            C["no_face_found_rate_on_real_faces"] = float(np.mean([score_of(r, det) == -1.0 for r in wiki]))
        nf = {}
        for src in ("ctrl_mei", "korus", "cocoglide", "openfake"):
            neg = [r for r in by_src.get(src, []) if int(r["m"]["label"]) == 0]
            if neg:
                s = np.array([score_of(r, det) for r in neg])
                nf[src] = {"n": int(s.size), "faces_found_fraction": float((s >= 0).mean()),
                           "fpr_at_high": float((s >= th[det]["high"]).mean())}
        C["false_positive_on_other_real_images"] = nf
        out["tasks"]["face_deepfake"] = C

    # ---------------- product level: the verdict the user sees ----------------
    for src, rs in sorted(by_src.items()):
        lab = np.array([int(r["m"]["label"]) for r in rs])
        dec = np.array([ORDER.get(r["r"]["overall_decision"], 0) for r in rs])
        blk = {"n": int(len(rs)), "n_manipulated_or_generated": int(lab.sum()), "n_real": int((lab == 0).sum())}
        if (lab == 0).any():
            blk["real_flagged_suspicious"] = float((dec[lab == 0] >= 3).mean())
            blk["real_needs_review_or_more"] = float((dec[lab == 0] >= 2).mean())
            blk["real_no_clear_sign"] = float((dec[lab == 0] == 1).mean())
        if (lab == 1).any():
            blk["fake_flagged_suspicious"] = float((dec[lab == 1] >= 3).mean())
            blk["fake_needs_review_or_more"] = float((dec[lab == 1] >= 2).mean())
            blk["fake_missed_no_clear_sign"] = float((dec[lab == 1] == 1).mean())
        trig = defaultdict(int)
        for r in rs:
            if r["r"]["overall_decision"] == "suspicious":
                for k, d in r["r"]["detectors"].items():
                    if d["decision"] == "suspicious":
                        trig[k] += 1
        blk["suspicious_triggered_by"] = dict(trig)
        secs = [r["r"].get("seconds_total") for r in rs if r["r"].get("seconds_total") is not None]
        if secs:
            blk["seconds_p50_p95"] = [float(np.percentile(secs, 50)), float(np.percentile(secs, 95))]
        out["product_level"][src] = blk
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"n_images": out["n_images"], "errors": len(errors), "counts": out["counts_by_source"]}, indent=1))


if __name__ == "__main__":
    main()
