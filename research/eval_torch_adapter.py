"""Evaluate a research (PyTorch) localization adapter over a manifest; store map-derived image scores + pixel metrics."""
import argparse, json, sys, time
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parent)); sys.path.insert(0, str(Path(__file__).resolve().parent / "adapters"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ifd.image_io import read_image
from eval_policies import pix_stats


def derived(heat):
    flat = np.sort(heat.ravel())[::-1]
    n = flat.size
    return {"max": float(flat[0]), "p999": float(flat[max(0, int(n * 0.001) - 1)]), "p99": float(flat[max(0, int(n * 0.01) - 1)]),
            "top1pct_mean": float(flat[: max(1, n // 100)].mean()), "mean": float(flat.mean()), "frac05": float((heat >= 0.5).mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest", type=Path); ap.add_argument("--adapter", required=True); ap.add_argument("--ckpt", required=True)
    ap.add_argument("--variant", default=""); ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--split", default="all"); ap.add_argument("--threads", type=int, default=2); ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()
    if a.adapter == "mesorch":
        from mesorch_torch import MesorchRunner
        runner = MesorchRunner(a.ckpt, a.variant or "mesorch", a.threads)
    elif a.adapter == "catnet":
        from catnet_torch import CatNetRunner
        runner = CatNetRunner(a.ckpt, a.threads)
    else:
        raise SystemExit("unknown adapter")
    root = a.manifest.parent
    rows = [json.loads(l) for l in open(a.manifest)]
    rows = [r for r in rows if a.split == "all" or r.get("split") == a.split]
    if a.limit: rows = rows[: a.limit]
    done = {json.loads(l)["image_id"] for l in open(a.out)} if a.out.exists() else set()
    with open(a.out, "a") as f:
        for i, r in enumerate(rows):
            if r["image_id"] in done: continue
            rec = {"image_id": r["image_id"], "label": r["label"], "split": r.get("split"), "detector": a.adapter + (":" + a.variant if a.variant else "")}
            try:
                path = root / r["image"]
                img = read_image(path, max_file_mb=200, max_pixels=60_000_000, min_side=8)
                t0 = time.perf_counter()
                heat = runner.predict(img.rgb, path=path)[0] if a.adapter == "catnet" else runner.predict(img.rgb)[0]
                rec["seconds"] = time.perf_counter() - t0
                rec["derived"] = derived(heat)
                if r.get("mask"):
                    mask = np.asarray(Image.open(root / r["mask"]).convert("L")) >= 128
                    rec["pixel"] = pix_stats(mask, heat) if mask.shape == heat.shape else {"error": "shape"}
                elif r["label"] == 0:
                    rec["pixel"] = pix_stats(np.zeros(heat.shape, bool), heat)
            except Exception as exc:
                rec["error"] = f"{type(exc).__name__}: {exc}"
            f.write(json.dumps(rec) + "\n"); f.flush()
            if i % 25 == 0: print(i, len(rows), r["image_id"], rec.get("derived", {}).get("max"), flush=True)


if __name__ == "__main__":
    main()
