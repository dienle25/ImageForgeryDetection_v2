"""Evaluate an image-level research adapter (PyTorch) over a manifest; resumable JSONL output.

Optional deterministic stratified subset: --subset-per-stratum N --stratum-field generator
(positives stratified by `stratum-field`, negatives by `neg-stratum-field`), seed fixed.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "adapters"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ifd.image_io import read_image  # noqa: E402


def pick_subset(rows, per_pos, per_neg, pos_field, neg_field, seed):
    rng = random.Random(seed)
    strata = defaultdict(list)
    for r in rows:
        key = ("pos", str(r.get(pos_field))) if r["label"] == 1 else ("neg", str(r.get(neg_field)))
        strata[key].append(r)
    out = []
    for key in sorted(strata):
        items = sorted(strata[key], key=lambda r: r["image_id"])
        rng.shuffle(items)
        out += items[: (per_pos if key[0] == "pos" else per_neg)]
    return sorted(out, key=lambda r: r["image_id"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--adapter", required=True, choices=["spai"])
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="all")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--per-pos", type=int, default=0)
    ap.add_argument("--per-neg", type=int, default=0)
    ap.add_argument("--pos-field", default="generator")
    ap.add_argument("--neg-field", default="real_source")
    ap.add_argument("--seed", type=int, default=20261012)
    ap.add_argument("--smallest-max-size", type=int, default=None)
    a = ap.parse_args()

    if a.adapter == "spai":
        from spai_torch import SpaiRunner
        runner = SpaiRunner(a.ckpt, threads=a.threads, smallest_max_size=a.smallest_max_size)
    root = a.manifest.parent
    rows = [json.loads(l) for l in a.manifest.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if a.split == "all" or r.get("split") == a.split]
    if a.per_pos or a.per_neg:
        rows = pick_subset(rows, a.per_pos, a.per_neg, a.pos_field, a.neg_field, a.seed)
    print("rows", len(rows), flush=True)
    done = set()
    if a.out.exists():
        done = {json.loads(l)["image_id"] for l in a.out.read_text().splitlines() if l.strip()}
    with a.out.open("a") as f:
        for i, r in enumerate(rows):
            if r["image_id"] in done:
                continue
            rec = {"image_id": r["image_id"], "label": r["label"], "split": r.get("split"), "detector": a.adapter}
            try:
                img = read_image(root / r["image"], max_file_mb=200, max_pixels=60_000_000, min_side=8)
                t0 = time.perf_counter()
                out = runner.predict(img.rgb)
                rec["seconds"] = time.perf_counter() - t0
                rec["score"] = out.pop("score")
                rec["details"] = out
            except Exception as exc:  # recorded, never silently dropped
                rec["error"] = f"{type(exc).__name__}: {exc}"
            f.write(json.dumps(rec) + "\n")
            f.flush()
            print(i, len(rows), r["image_id"], rec.get("score"), round(rec.get("seconds", 0), 1), flush=True)


if __name__ == "__main__":
    main()
