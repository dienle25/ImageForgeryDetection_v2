"""Evaluate face-deepfake detector on a manifest: per image = max p_fake over detected faces (largest 3)."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parent / "adapters"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ifd.image_io import read_image
from face_pipeline import FaceFinder, square_crop
from yermandy_torch import YermandyRunner

ap = argparse.ArgumentParser(); ap.add_argument("manifest", type=Path); ap.add_argument("--model", required=True)
ap.add_argument("--out", type=Path, required=True); ap.add_argument("--split", default="all"); ap.add_argument("--fp32", action="store_true")
a = ap.parse_args()
import torch
runner = YermandyRunner(a.model, 2, torch.float32 if a.fp32 else torch.bfloat16)
ff = FaceFinder()
rows = [json.loads(l) for l in open(a.manifest)]
rows = [r for r in rows if a.split == "all" or r.get("split") == a.split]
done = {json.loads(l)["image_id"] for l in open(a.out)} if a.out.exists() else set()
with open(a.out, "a") as f:
    for i, r in enumerate(rows):
        if r["image_id"] in done: continue
        rec = {"image_id": r["image_id"], "label": r["label"], "split": r.get("split")}
        try:
            img = read_image(a.manifest.parent / r["image"], max_file_mb=200, max_pixels=60_000_000, min_side=8)
            t0 = time.perf_counter()
            faces = ff.find(img.rgb)[:3]
            rec["n_faces"] = len(faces)
            if faces:
                crops = [square_crop(img.rgb, fc["box"])[0] for fc in faces]
                p, lg = runner.p_fake(crops)
                rec["face_scores"] = [float(x) for x in p]; rec["score"] = float(p.max())
                rec["face_det_scores"] = [fc["score"] for fc in faces]
            else:
                rec["score"] = None
            rec["seconds"] = time.perf_counter() - t0
        except Exception as exc:
            rec["error"] = f"{type(exc).__name__}: {exc}"
        f.write(json.dumps(rec) + "\n"); f.flush()
        if i % 50 == 0: print(i, len(rows), r["image_id"], rec.get("score"), flush=True)
