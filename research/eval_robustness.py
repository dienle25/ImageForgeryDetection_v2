"""Robustness of ONE shipped detector to benign post-processing applied to final-test images (no re-tuning).

Transforms (applied in memory, then re-decoded exactly like an upload):
  jpeg75          re-encode JPEG quality 75
  half_jpeg85     downscale x0.5 (Pillow bicubic) + JPEG 85   (typical messaging/social-media path)
  long1024_jpeg80 longest side 1024 + JPEG 80
  jpeg95          re-encode JPEG quality 95 (control for the two global adjustments below)
  bright115_jpeg95   global brightness x1.15 (Pillow ImageEnhance) + JPEG 95   (benign whole-image edit)
  contrast085_jpeg95 global contrast x0.85 (Pillow ImageEnhance) + JPEG 95     (benign whole-image edit)
  crop80_jpeg95      centre crop to 80 % of width and height + JPEG 95             (benign crop)
  rot90_jpeg95       pixels rotated 90 deg (no EXIF tag) + JPEG 95                 (benign rotation)

  python research/eval_robustness.py --config config.json --detector commfor_aigen \
      --manifest openfake/manifest.jsonl --split test --out robust_commfor.jsonl
"""
from __future__ import annotations

import argparse
import io
import json
import sys
import time
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ifd.analysis import Analyzer  # noqa: E402
from ifd.config import Config, load_config  # noqa: E402
from ifd.image_io import read_image  # noqa: E402


def transform(raw: bytes, name: str) -> bytes:
    im = Image.open(io.BytesIO(raw))
    im.load()
    from PIL import ImageOps
    im = ImageOps.exif_transpose(im).convert("RGB")
    if name == "half_jpeg85":
        im = im.resize((max(16, im.width // 2), max(16, im.height // 2)), Image.Resampling.BICUBIC)
        q = 85
    elif name == "long1024_jpeg80":
        s = 1024 / max(im.size)
        if s < 1:
            im = im.resize((max(16, round(im.width * s)), max(16, round(im.height * s))), Image.Resampling.BICUBIC)
        q = 80
    elif name == "jpeg75":
        q = 75
    elif name == "jpeg95":
        q = 95
    elif name == "bright115_jpeg95":
        from PIL import ImageEnhance
        im = ImageEnhance.Brightness(im).enhance(1.15)
        q = 95
    elif name == "contrast085_jpeg95":
        from PIL import ImageEnhance
        im = ImageEnhance.Contrast(im).enhance(0.85)
        q = 95
    elif name == "crop80_jpeg95":
        w, h = im.size
        cw, ch = max(16, round(w * 0.8)), max(16, round(h * 0.8))
        left, top = (w - cw) // 2, (h - ch) // 2
        im = im.crop((left, top, left + cw, top + ch))
        q = 95
    elif name == "rot90_jpeg95":
        im = im.transpose(Image.Transpose.ROTATE_90)
        q = 95
    else:
        raise ValueError(name)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=q)
    return buf.getvalue()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--detector", required=True)
    ap.add_argument("--manifest", type=Path, action="append", required=True)
    ap.add_argument("--split", action="append", required=True)
    ap.add_argument("--transforms", nargs="+", default=["jpeg75", "half_jpeg85", "long1024_jpeg80"])
    ap.add_argument("--label", type=int, default=None, help="only images with this label (0 = authentic/real)")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    data = load_config(a.config).to_json()
    for k, d in data["detectors"].items():
        d["enabled"] = (k == a.detector)
    cfg = Config(data, a.config.resolve().parent)
    an = Analyzer(cfg)
    det = an.detectors[a.detector]
    settings = cfg.detector(a.detector)
    lim = cfg.limits
    rows, seen = [], set()
    for m in a.manifest:
        for l in m.read_text(encoding="utf-8").splitlines():
            if l.strip():
                r = json.loads(l)
                if r["image_id"] in seen or r.get("split") not in a.split or r.get("exclude"):
                    continue
                if a.label is not None and int(r["label"]) != a.label:
                    continue
                seen.add(r["image_id"])
                r["_root"] = str(m.parent)
                rows.append(r)
    done = set()
    if a.out.exists():
        done = {(json.loads(l)["image_id"], json.loads(l)["transform"]) for l in a.out.read_text().splitlines() if l.strip()}
    with a.out.open("a") as f:
        for i, r in enumerate(rows):
            raw = (Path(r["_root"]) / r["image"]).read_bytes()
            for t in a.transforms:
                if (r["image_id"], t) in done:
                    continue
                rec = {"image_id": r["image_id"], "label": int(r["label"]), "transform": t, "detector": a.detector}
                try:
                    img = read_image(transform(raw, t), max_file_mb=lim["max_file_mb"], max_pixels=lim["max_pixels"],
                                     min_side=lim["min_side"], name="x.jpg")
                    t0 = time.perf_counter()
                    res = det.predict(img.rgb, settings)
                    rec.update(score=res.score, decision=res.decision, seconds=time.perf_counter() - t0,
                               size_wh=img.metadata["display_size_wh"])
                except Exception as exc:
                    rec["error"] = f"{type(exc).__name__}: {exc}"
                f.write(json.dumps(rec) + "\n")
                f.flush()
            if i % 100 == 0:
                print(i, len(rows), flush=True)


if __name__ == "__main__":
    main()
