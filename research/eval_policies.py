"""Evaluate PSCC-Net preprocessing policies on a manifest (JSONL) split; writes per-image records."""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ifd_core.pscc_runtime import PSCCSession, baseline_policy, resize_policy  # noqa: E402

POLICIES = {
    "base512": lambda s, rgb: baseline_policy(s, rgb, 512),
    "r256_bicubic": lambda s, rgb: resize_policy(s, rgb, 256, "bicubic"),
    "r256_bilinear": lambda s, rgb: resize_policy(s, rgb, 256, "bilinear"),
    "r256_area": lambda s, rgb: resize_policy(s, rgb, 256, "area"),
    "r256_aspect_bicubic": lambda s, rgb: resize_policy(s, rgb, 256, "bicubic", keep_aspect=True),
    "r384_bicubic": lambda s, rgb: resize_policy(s, rgb, 384, "bicubic"),
    "r512_bicubic": lambda s, rgb: resize_policy(s, rgb, 512, "bicubic"),
}


def read_rgb(path: Path) -> np.ndarray:
    im = Image.open(path)
    im = ImageOps.exif_transpose(im)
    if im.mode in ("RGBA", "LA") or "transparency" in im.info:
        rgba = im.convert("RGBA")
        bg = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, rgba)
    return np.array(im.convert("RGB"))


def pixel_auc(mask: np.ndarray, heat: np.ndarray) -> float | None:
    pos = heat[mask]
    neg = heat[~mask]
    if pos.size == 0 or neg.size == 0:
        return None
    # rank-based AUC (Mann-Whitney) on a subsample for speed
    rng = np.random.default_rng(0)
    if pos.size > 20000:
        pos = rng.choice(pos, 20000, replace=False)
    if neg.size > 50000:
        neg = rng.choice(neg, 50000, replace=False)
    allv = np.concatenate([pos, neg])
    order = allv.argsort(kind="mergesort")
    ranks = np.empty(len(allv))
    sv = allv[order]
    # average ranks for ties
    i = 0
    r = np.arange(1, len(allv) + 1, dtype=np.float64)
    uniq, idx_start, counts = np.unique(sv, return_index=True, return_counts=True)
    avg = idx_start + (counts + 1) / 2.0
    ranks[order] = np.repeat(avg, counts)
    return float((ranks[: len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def pix_stats(mask: np.ndarray, heat: np.ndarray, t: float = 0.5) -> dict:
    pred = heat >= t
    tp = int((pred & mask).sum()); fp = int((pred & ~mask).sum()); fn = int((~pred & mask).sum())
    return {"tp": tp, "fp": fp, "fn": fn, "frac_pred": float(pred.mean()),
            "f1": (2 * tp / (2 * tp + fp + fn)) if (2 * tp + fp + fn) else None,
            "iou": (tp / (tp + fp + fn)) if (tp + fp + fn) else None,
            "auc": pixel_auc(mask, heat), "heat_mean": float(heat.mean()), "heat_max": float(heat.max())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--split", default="dev")
    ap.add_argument("--policies", default="base512,r256_bicubic,r256_aspect_bicubic,r256_area")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--onnx", default=str(Path(__file__).resolve().parents[1] / "models/pscc_net.onnx"))
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    root = a.manifest.parent
    rows = [json.loads(l) for l in open(a.manifest)]
    rows = [r for r in rows if a.split in ("all", r["split"])]
    if a.limit:
        rows = rows[: a.limit]
    sess = PSCCSession(a.onnx, a.threads)
    pols = a.policies.split(",")
    done = set()
    if a.out.exists():
        done = {json.loads(l)["image_id"] for l in open(a.out)}
    with open(a.out, "a") as f:
        for i, r in enumerate(rows):
            if r["image_id"] in done:
                continue
            rgb = read_rgb(root / r["image"])
            mask = None
            if r.get("mask"):
                mask = np.asarray(Image.open(root / r["mask"]).convert("L")) >= 128
                if mask.shape != rgb.shape[:2]:
                    raise SystemExit(f"mask/image size mismatch for {r['image_id']}: {mask.shape} vs {rgb.shape}")
            rec = {"image_id": r["image_id"], "label": r["label"], "split": r["split"]}
            for p in pols:
                t0 = time.perf_counter()
                score, heat, info = POLICIES[p](sess, rgb)
                dt = time.perf_counter() - t0
                rec[p] = {"score": score, "seconds": dt, "logits": info.get("logits")}
                if mask is not None:
                    rec[p]["pixel"] = pix_stats(mask, heat)
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 20 == 0:
                print(i, len(rows), r["image_id"], {p: round(rec[p]["score"], 4) for p in pols}, flush=True)


if __name__ == "__main__":
    main()
