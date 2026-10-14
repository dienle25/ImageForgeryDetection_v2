"""Build JSONL manifests (common schema) for external public datasets used ONLY for evaluation."""
from __future__ import annotations

import csv
import hashlib
import io
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps


def decoded_hash(path: Path) -> tuple[str, int, int]:
    im = ImageOps.exif_transpose(Image.open(path)).convert("RGB")
    a = np.asarray(im)
    h = hashlib.sha256()
    h.update(np.asarray(a.shape, dtype="<i8").tobytes()); h.update(a.tobytes())
    return h.hexdigest(), a.shape[1], a.shape[0]


def row(root: Path, image: Path, **kw) -> dict:
    raw = image.read_bytes()
    dh, w, h = decoded_hash(image)
    fmt = Image.open(image).format
    base = {"image": str(image.relative_to(root)), "sha256": hashlib.sha256(raw).hexdigest(),
            "decoded_rgb_sha256": dh, "width": w, "height": h, "encoding": fmt}
    base.update(kw)
    return base


def cocoglide(root: Path) -> list[dict]:
    rows = []
    with open(root / "table.csv", newline="") as f:
        for r in csv.DictReader(f):
            cid = r["real"].rsplit("_", 1)[1].split(".")[0]
            grp = f"coco_{cid}"
            common = dict(task="local_manipulation", group_id=grp, source_id=grp, donor_ids=[], split="test_cocoglide",
                          dataset="CocoGlide", license_source="COCO images (Flickr licences, many CC BY-NC*); annotations CC BY 4.0; dataset by GRIP-UNINA - research evaluation only",
                          post_processing="as distributed (PNG 256x256)")
            rows.append(row(root, root / r["real"], image_id=f"cocoglide_real_{cid}", label=0, operation="authentic",
                            mask=None, label_provenance="CocoGlide 'real' counterpart (COCO val2017 crop)", **common))
            m = np.asarray(Image.open(root / r["mask"]).convert("L")) > 127
            rows.append(row(root, root / r["fake"], image_id=f"cocoglide_fake_{cid}", label=1, operation="ai_inpainting_glide",
                            mask=r["mask"], edited_fraction=float(m.mean()), prompt=r["prompt"],
                            label_provenance="CocoGlide: GLIDE text-guided inpainting of a COCO object mask", **common))
    return rows


def korus(root: Path) -> list[dict]:
    """Korus & Huang Realistic Tampering: 4 cameras x 55 pristine/tampered pairs (TIFF 1920x1080); 3-level GT (gray+white = tampered)."""
    rows = []
    for cam_dir in sorted((root / "data-images").iterdir()):
        if not (cam_dir / "pristine").is_dir():
            continue
        cam = cam_dir.name
        for tif in sorted((cam_dir / "pristine").glob("*.TIF")):
            stem = tif.stem
            grp = f"korus_{cam}_{stem}"
            common = dict(task="local_manipulation", group_id=grp, source_id=grp, donor_ids=[], split="test_korus", dataset="Korus Realistic Tampering",
                          camera=cam, license_source="P. Korus & J. Huang - 'educational and research purposes only' - evaluation only, not redistributed",
                          post_processing="as distributed (uncompressed TIFF)")
            rows.append(row(root, tif, image_id=f"korus_pristine_{cam}_{stem}", label=0, operation="authentic", mask=None,
                            label_provenance="Korus pristine camera image", **common))
            gt_src = cam_dir / "ground-truth" / f"{stem}.PNG"
            g = np.asarray(Image.open(gt_src).convert("L"))
            binm = (g > 0).astype(np.uint8) * 255   # gray (collateral) + white (semantic) = tampered, as the dataset page recommends
            mdir = root / "masks_binary"; mdir.mkdir(exist_ok=True)
            mpath = mdir / f"{cam}_{stem}.png"
            Image.fromarray(binm).save(mpath)
            rows.append(row(root, cam_dir / "tampered-realistic" / f"{stem}.TIF", image_id=f"korus_tampered_{cam}_{stem}", label=1,
                            operation="realistic_hand_made", mask=str(mpath.relative_to(root)), edited_fraction=float((binm > 0).mean()),
                            label_provenance="Korus hand-made realistic forgery (object insertion/removal); GT gray+white binarized", **common))
    return rows


if __name__ == "__main__":
    name, root, out = sys.argv[1], Path(sys.argv[2]), Path(sys.argv[3])
    rows = {"cocoglide": cocoglide, "korus": korus}[name](root)
    with open(out, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    print(name, len(rows), "rows ->", out)
