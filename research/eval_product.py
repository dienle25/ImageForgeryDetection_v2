"""Run the SHIPPED analysis pipeline (ifd.analysis.Analyzer + config.json + ONNX models) over JSONL manifests.

One record per image (resumable): every enabled detector's score/decision/details, overall decision,
pixel metrics for heatmap detectors (vs GT mask, or false activation on authentic images), timings.
This is the code path users run (same read_image limits, same preprocessing, same thresholds).

  python research/eval_product.py --config config.json \
      --manifest M.jsonl [--manifest M2.jsonl] --split dev --out runs/dev_product.jsonl
"""
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
from ifd.analysis import Analyzer  # noqa: E402
from ifd.config import load_config  # noqa: E402
from ifd.errors import ForgeryError  # noqa: E402
from eval_policies import pix_stats  # noqa: E402

KEEP_DETAILS = ("logit", "logits_authentic_forged", "native_map_stats", "heatmap_stats", "faces_found",
                "faces_checked", "faces", "image_score_statistic")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--manifest", type=Path, action="append", required=True)
    ap.add_argument("--split", action="append", default=None, help="keep rows whose split is one of these")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--only-label", type=int, default=None)
    a = ap.parse_args()

    cfg = load_config(a.config)
    if a.threads is not None:
        cfg.data.setdefault("runtime", {})["threads"] = a.threads
    t0 = time.perf_counter()
    analyzer = Analyzer(cfg)
    print(f"loaded {list(analyzer.detectors)} in {time.perf_counter() - t0:.1f}s", flush=True)

    rows, seen = [], set()
    for m in a.manifest:
        for line in m.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r["image_id"] in seen:
                continue
            if a.split and r.get("split") not in a.split:
                continue
            if a.only_label is not None and int(r["label"]) != a.only_label:
                continue
            if r.get("exclude"):
                continue
            seen.add(r["image_id"])
            r["_root"] = str(m.parent)
            rows.append(r)
    done = set()
    if a.out.exists():
        done = {json.loads(l)["image_id"] for l in a.out.read_text().splitlines() if l.strip()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    print("rows", len(rows), "done", len(done), flush=True)
    with a.out.open("a") as f:
        for i, r in enumerate(rows):
            if r["image_id"] in done:
                continue
            root = Path(r["_root"])
            rec = {"image_id": r["image_id"], "label": int(r["label"]), "split": r.get("split")}
            try:
                t1 = time.perf_counter()
                rep = analyzer.analyze(root / r["image"], name=Path(r["image"]).name)
                rec["seconds_total"] = time.perf_counter() - t1
                rec["overall_decision"] = rep.summary["overall_decision"]
                rec["tasks_checked"] = [t["task"] for t in rep.summary["tasks_checked"]]
                rec["detectors"] = {}
                mask = None
                if r.get("mask"):
                    mask = np.asarray(Image.open(root / r["mask"]).convert("L")) >= 128
                for res in rep.results:
                    d = {"task": res.task, "score": res.score, "decision": res.decision,
                         "seconds": next(iter(res.timing.values()), None),
                         "details": {k: v for k, v in res.details.items() if k in KEEP_DETAILS}}
                    if res.heatmap is not None:
                        pt = float(res.thresholds.get("pixel", 0.5))
                        if mask is not None:
                            d["pixel"] = (pix_stats(mask, res.heatmap, pt) if mask.shape == res.heatmap.shape
                                          else {"error": f"mask shape {mask.shape} vs heatmap {res.heatmap.shape}"})
                        elif int(r["label"]) == 0:
                            d["pixel"] = pix_stats(np.zeros(res.heatmap.shape, bool), res.heatmap, pt)
                    rec["detectors"][res.key] = d
            except ForgeryError as exc:
                rec["error"] = f"ForgeryError: {exc}"
            except Exception as exc:  # recorded, never silently dropped
                rec["error"] = f"{type(exc).__name__}: {exc}"
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            if i % 25 == 0:
                sc = {k: (None if v["score"] is None else round(v["score"], 4)) for k, v in rec.get("detectors", {}).items()}
                print(i, len(rows), r["image_id"], sc, round(rec.get("seconds_total", 0), 2), flush=True)


if __name__ == "__main__":
    main()
