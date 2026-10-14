"""Run product detectors (ifd.detectors.*) over a JSONL manifest; write per-image records (resumable)."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from ifd.image_io import read_image  # noqa: E402
from eval_policies import pix_stats  # noqa: E402

TH = {"low": 0.35, "high": 0.65, "pixel": 0.5, "status": "evaluation only"}


def make(kind: str, model: str, policy: str | None, threads: int):
    if kind == "pscc":
        from ifd.detectors.pscc import PSCCDetector
        return PSCCDetector(model, {"policy": json.loads(policy), "thresholds": TH}, threads)
    if kind == "commfor":
        from ifd.detectors.commfor import CommForDetector
        return CommForDetector(model, {"thresholds": TH}, threads)
    raise SystemExit(f"unknown detector kind {kind}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path)
    ap.add_argument("--kind", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--policy")
    ap.add_argument("--name", required=True, help="record key for this detector/config")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="all")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-pixels", type=int, default=60_000_000)
    a = ap.parse_args()
    det = make(a.kind, a.model, a.policy, a.threads)
    root = a.manifest.parent
    rows = [json.loads(l) for l in open(a.manifest)]
    rows = [r for r in rows if a.split == "all" or r.get("split") == a.split]
    if a.limit:
        rows = rows[: a.limit]
    done = set()
    if a.out.exists():
        done = {json.loads(l)["image_id"] for l in open(a.out)}
    with open(a.out, "a") as f:
        for i, r in enumerate(rows):
            if r["image_id"] in done:
                continue
            rec = {"image_id": r["image_id"], "label": r["label"], "split": r.get("split"), "detector": a.name}
            try:
                img = read_image(root / r["image"], max_file_mb=200, max_pixels=a.max_pixels, min_side=8)
                t0 = time.perf_counter()
                res = det.predict(img.rgb)
                rec["seconds"] = time.perf_counter() - t0
                rec["score"] = res.score
                rec["details"] = {k: v for k, v in res.details.items() if k in ("logit", "logits_authentic_forged", "heatmap_stats")}
                if res.heatmap is not None and r.get("mask"):
                    mask = np.asarray(Image.open(root / r["mask"]).convert("L")) >= 128
                    if mask.shape != res.heatmap.shape:
                        rec["mask_error"] = f"shape {mask.shape} vs {res.heatmap.shape}"
                    else:
                        rec["pixel"] = pix_stats(mask, res.heatmap)
                elif res.heatmap is not None and r["label"] == 0:
                    rec["pixel"] = pix_stats(np.zeros(res.heatmap.shape, bool), res.heatmap)
            except Exception as exc:
                rec["error"] = f"{type(exc).__name__}: {exc}"
            f.write(json.dumps(rec) + "\n")
            f.flush()
            if i % 25 == 0:
                print(i, len(rows), r["image_id"], rec.get("score"), flush=True)


if __name__ == "__main__":
    main()
