"""Generate a controlled, lineage-tracked manipulation benchmark from pristine camera JPEGs.

Every edited image has a paired authentic image that went through the SAME post-processing
(save format / JPEG quality / resize), so compression or resizing cannot act as a label shortcut.
Edits are scripted (alpha-feathered splice, transformed copy-move, OpenCV TELEA inpainting removal);
they are NOT natural or professional forgeries and NOT neural inpainting.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageOps

POST = [  # (name, kind, params)
    ("jpeg95", "jpeg", {"quality": 95}),
    ("jpeg85", "jpeg", {"quality": 85}),
    ("jpeg75", "jpeg", {"quality": 75}),
    ("png", "png", {}),
    ("half_jpeg85", "resize_jpeg", {"scale": 0.5, "quality": 85}),
    ("long1024_jpeg90", "long_jpeg", {"long": 1024, "quality": 90}),
]
SIZES = {"small": (0.003, 0.015), "medium": (0.015, 0.06), "large": (0.06, 0.20)}
OPS = ["splice", "copymove", "removal"]


def load_working(path: Path, long_side: int) -> Image.Image:
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    s = long_side / max(im.size)
    if s < 1:
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=92)  # benign "web copy" of the camera photo
    return Image.open(io.BytesIO(buf.getvalue())).convert("RGB")


def blob_mask(rng, h, w, frac):
    """Random smooth blob (ellipse / star-polygon / rounded rectangle) with target area fraction."""
    target = frac * h * w
    kind = rng.choice(["ellipse", "poly", "rect"])
    for _ in range(50):
        m = np.zeros((h, w), np.uint8)
        r = math.sqrt(target / math.pi) * rng.uniform(0.9, 1.1)
        ar = rng.uniform(0.5, 2.0)
        rx, ry = r * math.sqrt(ar), r / math.sqrt(ar)
        cx = rng.uniform(rx + 2, w - rx - 2) if w > 2 * rx + 4 else w / 2
        cy = rng.uniform(ry + 2, h - ry - 2) if h > 2 * ry + 4 else h / 2
        if kind == "ellipse":
            cv2.ellipse(m, (int(cx), int(cy)), (max(2, int(rx)), max(2, int(ry))), float(rng.uniform(0, 180)), 0, 360, 1, -1)
        elif kind == "rect":
            x0, y0 = int(cx - rx), int(cy - ry)
            cv2.rectangle(m, (max(0, x0), max(0, y0)), (min(w - 1, int(cx + rx)), min(h - 1, int(cy + ry))), 1, -1)
        else:
            n = int(rng.integers(6, 12))
            ang = np.sort(rng.uniform(0, 2 * math.pi, n))
            rad = rng.uniform(0.6, 1.25, n)
            pts = np.stack([cx + rx * rad * np.cos(ang), cy + ry * rad * np.sin(ang)], 1).astype(np.int32)
            cv2.fillPoly(m, [pts], 1)
            m = cv2.GaussianBlur(m.astype(np.float32), (0, 0), max(1.0, r / 8)) > 0.5
            m = m.astype(np.uint8)
        area = m.sum()
        if area > 0 and 0.6 * target <= area <= 1.5 * target:
            return m.astype(bool), kind
    return m.astype(bool), kind


def bbox(m):
    ys, xs = np.nonzero(m)
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1


def paste(dst: np.ndarray, patch: np.ndarray, pmask: np.ndarray, oy: int, ox: int, feather: float):
    h, w = pmask.shape
    alpha = pmask.astype(np.float32)
    if feather > 0:
        alpha = cv2.GaussianBlur(alpha, (0, 0), feather)
    region = dst[oy:oy + h, ox:ox + w].astype(np.float32)
    out = region * (1 - alpha[..., None]) + patch.astype(np.float32) * alpha[..., None]
    dst[oy:oy + h, ox:ox + w] = np.clip(np.round(out), 0, 255).astype(np.uint8)
    full = np.zeros(dst.shape[:2], bool)
    full[oy:oy + h, ox:ox + w] = alpha > 0.5
    return full


def do_edit(rng, src: np.ndarray, donor: np.ndarray | None, op: str, frac: float):
    h, w = src.shape[:2]
    m, shape = blob_mask(rng, h, w, frac)
    y0, y1, x0, x1 = bbox(m)
    pm = m[y0:y1, x0:x1]
    ph, pw = pm.shape
    out = src.copy()
    info = {"shape": shape}
    feather = float(rng.choice([0.0, 1.0, 2.0]))
    if op == "removal":
        dil = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8))
        out = cv2.inpaint(src, dil, 5, cv2.INPAINT_TELEA)
        gt = dil.astype(bool)
        info.update(method="opencv_inpaint_telea_r5")
        return out, gt, None, info
    if op == "splice":
        dh, dw = donor.shape[:2]
        sc = float(rng.uniform(0.8, 1.2))
        need_h, need_w = int(math.ceil(ph / sc)), int(math.ceil(pw / sc))
        if need_h >= dh or need_w >= dw:
            sc = max(ph / (dh - 1), pw / (dw - 1)) * 1.05
            need_h, need_w = int(math.ceil(ph / sc)), int(math.ceil(pw / sc))
        sy, sx = int(rng.integers(0, dh - need_h)), int(rng.integers(0, dw - need_w))
        patch = cv2.resize(donor[sy:sy + need_h, sx:sx + need_w], (pw, ph), interpolation=cv2.INTER_LINEAR)
        gt = paste(out, patch, pm, y0, x0, feather)
        info.update(donor_box=[sy, sx, need_h, need_w], donor_scale=sc, feather=feather)
        return out, gt, None, info
    # copymove: take patch from another location in the same image, transform, paste at blob location
    ang = float(rng.uniform(-25, 25)) if rng.random() < 0.5 else 0.0
    sc = float(rng.uniform(0.85, 1.15)) if rng.random() < 0.5 else 1.0
    for _ in range(100):
        sy, sx = int(rng.integers(0, max(1, h - ph))), int(rng.integers(0, max(1, w - pw)))
        if abs(sy - y0) > ph * 0.6 or abs(sx - x0) > pw * 0.6:
            break
    srcpatch = src[sy:sy + ph, sx:sx + pw]
    if srcpatch.shape[:2] != (ph, pw):
        srcpatch = cv2.resize(srcpatch, (pw, ph))
    M = cv2.getRotationMatrix2D((pw / 2, ph / 2), ang, sc)
    patch = cv2.warpAffine(srcpatch, M, (pw, ph), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
    gt = paste(out, patch, pm, y0, x0, feather)
    srcmask = np.zeros((h, w), bool)
    srcmask[sy:sy + ph, sx:sx + pw] = pm
    info.update(source_box=[sy, sx, ph, pw], rotation_deg=ang, scale=sc, feather=feather)
    return out, gt, srcmask, info


def post_process(rgb: np.ndarray, post, masks):
    name, kind, p = post
    im = Image.fromarray(rgb)
    out_masks = list(masks)
    if kind == "resize_jpeg" or kind == "long_jpeg":
        s = p["scale"] if kind == "resize_jpeg" else min(1.0, p["long"] / max(im.size))
        size = (max(16, round(im.width * s)), max(16, round(im.height * s)))
        if size != im.size:
            im = im.resize(size, Image.Resampling.BICUBIC)
            out_masks = [None if m is None else np.asarray(Image.fromarray(m.astype(np.uint8) * 255).resize(size, Image.Resampling.BILINEAR)) >= 128 for m in masks]
    buf = io.BytesIO()
    if kind == "png":
        im.save(buf, format="PNG")
        ext = "png"
    else:
        im.save(buf, format="JPEG", quality=p["quality"])
        ext = "jpg"
    return buf.getvalue(), ext, out_masks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--raw-root", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--long-side", type=int, default=1536)
    ap.add_argument("--pairs-per-source", type=int, default=2)
    a = ap.parse_args()
    pool = json.load(open(a.pool))
    rng = np.random.default_rng(a.seed)
    groups = sorted({r["near_dup_group"] for r in pool})
    rng.shuffle(groups)
    split_of = {g: ("dev" if i % 2 == 0 else "test") for i, g in enumerate(groups)}
    (a.out / "images").mkdir(parents=True, exist_ok=True)
    (a.out / "masks").mkdir(parents=True, exist_ok=True)
    work = {}
    for r in pool:
        work[r["p"]] = r
    rows = []
    by_split = {"dev": [r for r in pool if split_of[r["near_dup_group"]] == "dev"],
                "test": [r for r in pool if split_of[r["near_dup_group"]] == "test"]}
    for split, members in by_split.items():
        for idx, r in enumerate(members):
            src_id = "mei_" + r["sha256"][:12]
            W = np.asarray(load_working(a.raw_root / r["p"], a.long_side))
            for k in range(a.pairs_per_source):
                op = OPS[(idx + k) % 3]
                size_name = list(SIZES)[(idx // 3 + k) % 3]
                post = POST[int(rng.integers(0, len(POST)))]
                frac = float(rng.uniform(*SIZES[size_name]))
                donor_row = None
                donor = None
                if op == "splice":
                    cands = [m for m in members if m["near_dup_group"] != r["near_dup_group"]]
                    donor_row = cands[int(rng.integers(0, len(cands)))]
                    donor = np.asarray(load_working(a.raw_root / donor_row["p"], a.long_side))
                edit_seed = int(rng.integers(0, 2**31 - 1))
                erng = np.random.default_rng(edit_seed)
                edited, gt, srcmask, info = do_edit(erng, W, donor, op, frac)
                for label, rgb, masks in ((0, W, [np.zeros(W.shape[:2], bool), None]), (1, edited, [gt, srcmask])):
                    raw, ext, pm = post_process(rgb, post, masks)
                    image_id = f"{src_id}_{k}_{'auth' if label == 0 else op}_{post[0]}"
                    ipath = a.out / "images" / f"{image_id}.{ext}"
                    ipath.write_bytes(raw)
                    mpath = a.out / "masks" / f"{image_id}_mask.png"
                    Image.fromarray(pm[0].astype(np.uint8) * 255).save(mpath)
                    spath = None
                    if label == 1 and pm[1] is not None:
                        spath = a.out / "masks" / f"{image_id}_cmsource.png"
                        Image.fromarray(pm[1].astype(np.uint8) * 255).save(spath)
                    dec = np.asarray(Image.open(io.BytesIO(raw)).convert("RGB"))
                    rows.append({
                        "image_id": image_id, "image": str(ipath.relative_to(a.out)), "mask": str(mpath.relative_to(a.out)),
                        "copymove_source_mask": str(spath.relative_to(a.out)) if spath else None,
                        "label": label, "task": "local_manipulation", "operation": "authentic" if label == 0 else op,
                        "size_bucket": "none" if label == 0 else size_name,
                        "edited_fraction": float(pm[0].mean()), "post_processing": post[0],
                        "encoding": "PNG" if ext == "png" else "JPEG",
                        "jpeg_quality": post[2].get("quality"), "width": int(dec.shape[1]), "height": int(dec.shape[0]),
                        "source_id": src_id, "source_file": r["p"], "camera": f"{r['make']} {r['model']}",
                        "donor_ids": ["mei_" + donor_row["sha256"][:12]] if (label == 1 and donor_row) else [],
                        "group_id": f"mei_grp_{r['near_dup_group']}", "split": split,
                        "pair_id": f"{src_id}_{k}", "edit_seed": edit_seed if label == 1 else None,
                        "edit_info": info if label == 1 else None,
                        "label_provenance": "scripted edit with exact mask" if label == 1 else
                            "camera JPEG (EXIF make/model, no editing-software tag), resized<=1536 + JPEG q92, then same post-processing as its edited pair",
                        "license_source": "drewnoakes/metadata-extractor-images: 'You are free to use these media files however you wish.'",
                        "sha256": hashlib.sha256(raw).hexdigest(),
                        "decoded_rgb_sha256": hashlib.sha256(dec.tobytes()).hexdigest(),
                    })
        print(split, len(members), "sources done")
    with open(a.out / "manifest.jsonl", "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")
    meta = {"seed": a.seed, "long_side": a.long_side, "post_processing": POST, "size_buckets": SIZES,
            "n_images": len(rows), "split_rule": "near-duplicate groups alternately assigned dev/test after seeded shuffle; donors drawn within the same split"}
    json.dump(meta, open(a.out / "generation.json", "w"), indent=1)
    print("images", len(rows))


if __name__ == "__main__":
    main()
