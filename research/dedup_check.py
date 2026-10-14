"""Leakage / duplicate audit across every manifest used for development and final tests.

For each image: file SHA-256 (from manifest), decoded-pixel SHA-256 (manifest or recomputed), and a 64-bit
difference hash (dHash, 9x8 grayscale, Pillow BILINEAR) for near-duplicates.
Reports:
  * exact byte / pixel duplicates that cross split or group boundaries (leakage)
  * near-duplicate pairs (Hamming distance <= --max-hamming) across DIFFERENT splits, and across datasets
Pairs inside the same group (e.g. an edit and its own original) are expected and listed only as counts.

  python research/dedup_check.py --manifest M1 --manifest M2 ... --out dedup_report.json
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def fingerprints(path: Path) -> tuple[int, str]:
    import hashlib
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        rgb = np.asarray(im.convert("RGB"))
        small = im.convert("L").resize((9, 8), Image.Resampling.BILINEAR)
        a = np.asarray(small, dtype=np.int16)
    h = hashlib.sha256()
    h.update(np.asarray(rgb.shape, dtype="<i8").tobytes())
    h.update(np.ascontiguousarray(rgb).tobytes())
    bits = (a[:, 1:] > a[:, :-1]).ravel()
    return int("".join("1" if b else "0" for b in bits), 2), h.hexdigest()


def popcount(x: int) -> int:
    return bin(x).count("1")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, action="append", required=True)
    ap.add_argument("--max-hamming", type=int, default=4)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cache", type=Path, default=None)
    a = ap.parse_args()
    cache = {}
    if a.cache and a.cache.exists():
        cache = json.loads(a.cache.read_text())
    rows = []
    seen = set()
    for m in a.manifest:
        ds = m.parent.name
        for line in m.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["image_id"] in seen:
                continue
            seen.add(r["image_id"])
            p = m.parent / r["image"]
            key = str(p)
            if key not in cache:
                try:
                    cache[key] = list(fingerprints(p))
                except Exception as exc:
                    cache[key] = [None, None]
                    print("unreadable", p, exc)
            dh, pix = cache[key]
            rows.append({"id": r["image_id"], "ds": ds, "split": r.get("split"), "group": r.get("group_id", r["image_id"]),
                         "label": r["label"], "sha": r.get("sha256"), "pix": r.get("decoded_rgb_sha256") or pix,
                         "dh": dh, "exclude": bool(r.get("exclude"))})
    if a.cache:
        a.cache.write_text(json.dumps(cache))
    rep = {"images": len(rows), "datasets": sorted({r["ds"] for r in rows})}
    # exact duplicates
    for field in ("sha", "pix"):
        by = defaultdict(list)
        for r in rows:
            if r[field]:
                by[r[field]].append(r)
        cross = []
        within = 0
        for v in by.values():
            if len(v) < 2:
                continue
            splits = {x["split"] for x in v}
            groups = {x["group"] for x in v}
            if len(splits) > 1 or len(groups) > 1:
                cross.append([{k: x[k] for k in ("id", "ds", "split", "group", "label", "exclude")} for x in v])
            else:
                within += 1
        rep[f"exact_{field}_duplicate_sets_crossing_split_or_group"] = cross
        rep[f"exact_{field}_duplicate_sets_within_one_group"] = within
    # near duplicates (dHash): pigeonhole on 5 chunks (13,13,13,13,12 bits) -> any pair with <=4 differing bits
    # shares at least one identical chunk, so bucketing by chunk finds every such pair.
    valid = [r for r in rows if r["dh"] is not None]
    buckets = defaultdict(list)
    spans = [(0, 13), (13, 26), (26, 39), (39, 52), (52, 64)]
    for i, r in enumerate(valid):
        for c, (lo, hi) in enumerate(spans):
            buckets[(c, (r["dh"] >> lo) & ((1 << (hi - lo)) - 1))].append(i)
    pairs = set()
    for idx in buckets.values():
        if len(idx) < 2:
            continue
        for x in range(len(idx)):
            for y in range(x + 1, len(idx)):
                i, j = idx[x], idx[y]
                if popcount(valid[i]["dh"] ^ valid[j]["dh"]) <= a.max_hamming:
                    pairs.add((min(i, j), max(i, j)))
    same_group = cross_split = same_split_other_group = 0
    examples_cross, examples_same_split = [], []
    for i, j in sorted(pairs):
        ri, rj = valid[i], valid[j]
        if ri["group"] == rj["group"]:
            same_group += 1
            continue
        item = {"a": {k: ri[k] for k in ("id", "ds", "split", "label")}, "b": {k: rj[k] for k in ("id", "ds", "split", "label")},
                "hamming": popcount(ri["dh"] ^ rj["dh"])}
        if ri["split"] != rj["split"]:
            cross_split += 1
            examples_cross.append(item)
        else:
            same_split_other_group += 1
            examples_same_split.append(item)
    rep["near_duplicates"] = {"max_hamming": a.max_hamming, "pairs_total": len(pairs), "pairs_same_group": same_group,
                              "pairs_cross_split_different_group": cross_split,
                              "pairs_same_split_different_group": same_split_other_group,
                              "cross_split_examples": examples_cross[:200],
                              "same_split_examples": examples_same_split[:50],
                              "note": "dHash ignores colour and fine detail; flagged pairs need visual review "
                                      "(e.g. uniform/dark images collide)."}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(rep, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({k: (v if not isinstance(v, list) else len(v)) for k, v in rep.items() if k != "near_duplicates"}, indent=1))
    print(json.dumps({k: v for k, v in rep["near_duplicates"].items() if not k.endswith("examples")}, indent=1))


if __name__ == "__main__":
    main()
