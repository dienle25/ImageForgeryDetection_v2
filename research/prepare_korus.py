"""Korus Realistic Tampering: lossless TIFF->PNG copies + group split (P02 amendment, before any scoring).

- Pixels are preserved exactly: decoded RGB SHA-256 of every PNG must equal the TIFF's (checked).
- Split by scene pair (pristine + tampered share group_id), stratified by camera, seed 20261012:
  18 pairs per camera -> dev_korus, remaining 37 -> test_korus.
Usage: python research/prepare_korus.py RAW_KORUS_DIR OUT_DIR
"""
from __future__ import annotations

import hashlib
import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

SEED = 20261012
DEV_PER_CAMERA = 18


def decoded_hash(rgb: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(rgb.shape, dtype="<i8").tobytes())
    h.update(np.ascontiguousarray(rgb).tobytes())
    return h.hexdigest()


def main():
    raw, out = Path(sys.argv[1]), Path(sys.argv[2])
    rows = [json.loads(l) for l in (raw / "manifest.jsonl").read_text().splitlines() if l.strip()]
    by_cam = defaultdict(set)
    for r in rows:
        by_cam[r["camera"]].add(r["group_id"])
    rng = random.Random(SEED)
    dev_groups = set()
    for cam in sorted(by_cam):
        groups = sorted(by_cam[cam])
        rng.shuffle(groups)
        dev_groups.update(groups[:DEV_PER_CAMERA])
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)
    new_rows = []
    for r in rows:
        src = raw / r["image"]
        with Image.open(src) as im:
            im.load()
            rgb = np.asarray(im.convert("RGB"))
        dst_rel = Path("images") / (r["image_id"] + ".png")
        dst = out / dst_rel
        if not dst.exists():
            Image.fromarray(rgb).save(dst, format="PNG", compress_level=6)
        with Image.open(dst) as im2:
            back = np.asarray(im2.convert("RGB"))
        dh = decoded_hash(back)
        if dh != r["decoded_rgb_sha256"]:
            raise SystemExit(f"pixel mismatch after PNG conversion: {r['image_id']}")
        nr = dict(r)
        nr["image"] = str(dst_rel)
        nr["sha256"] = hashlib.sha256(dst.read_bytes()).hexdigest()
        nr["encoding"] = "PNG"
        nr["original_file"] = r["image"]
        nr["original_sha256"] = r["sha256"]
        nr["post_processing"] = "lossless TIFF->PNG copy (decoded pixels identical to distributed TIFF)"
        if r.get("mask"):
            mdst = Path("masks") / Path(r["mask"]).name
            if not (out / mdst).exists():
                shutil.copyfile(raw / r["mask"], out / mdst)
            nr["mask"] = str(mdst)
        nr["split"] = "dev_korus" if r["group_id"] in dev_groups else "test_korus"
        new_rows.append(nr)
    with (out / "manifest.jsonl").open("w") as f:
        for nr in new_rows:
            f.write(json.dumps(nr, ensure_ascii=False) + "\n")
    cnt = defaultdict(int)
    for nr in new_rows:
        cnt[(nr["split"], nr["label"])] += 1
    print(dict(cnt))


if __name__ == "__main__":
    main()
