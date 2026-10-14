"""Download the ORIGINAL upstream weights / evaluation data used in this project and verify their SHA-256.

Only needed to REPRODUCE the ONNX export or the evaluation; the shipped app never downloads anything.
Run on a machine with Internet; check each licence (LICENSES_VI.md) before use.

    python tools/fetch_sources.py --dest D:\\sources [--only weights|data] [--list]

Google Drive files use the direct-download endpoint; very large files may need a browser if Google adds a
virus-scan interstitial. Every file is verified against the size/SHA-256 observed when this project downloaded it.
"""
from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import urllib.request
from pathlib import Path

GD = "https://drive.usercontent.google.com/download?id={}&export=download&confirm=t"
SOURCES = [
    # kind, file name, url, bytes, sha256, note
    ("weights", "mesorch-98.pth", GD.format("1PJxKteinMyaAYokKy0JhuzBnBc6bGsau"), 1023886070,
     "6d8fcd7ce7616d819bec6a9ed461b27187101e67247f8b2d2483fdc1f25f685a", "Mesorch official checkpoint (README link of github.com/scu-zjz/Mesorch)"),
    ("weights", "commfor_model.safetensors", "https://huggingface.co/OwensLab/commfor-model-384/resolve/main/model.safetensors", 87262324,
     "b89f36275f3bf5e2b040eee36597a8f19db051bff9a473a9cf7b2466284fb387", "Community Forensics ViT-S/16@384"),
    ("weights", "yermandy_model.torchscript", "https://huggingface.co/yermandy/deepfake-detection/resolve/main/model.torchscript", 607308240,
     "55c54d3f4d301ce07f0d08126fc4202572d20ea6952b039bc7a949a43714157d", "CLIP ViT-L/14 LN-tuned deepfake detector"),
    ("weights", "face_detection_yunet_2023mar.onnx", "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx", 232589,
     "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4", "YuNet face detector (shipped unchanged)"),
    ("weights", "CAT_full_v2.pth.tar", GD.format("1tyOKVdx6UMys2OcNpUj9r6scxNIpcoLE"), 915503873,
     "f82aaafdd1142775231feedcea0bb7027f7370561d9e8d107465454001865989", "CAT-Net v2 (candidate, not shipped)"),
    ("weights", "spai.pth", GD.format("1vvXmZqs6TVJdj8iF1oJ4L_fcgdQrp_YI"), 934865338,
     "24159f27d7c8c2cd0cb6c4019189eb89ad0874a0d9d15f8dc9afd39ca9648a55", "SPAI (candidate, not shipped)"),
    ("data", "CocoGlide.zip", "https://www.grip.unina.it/download/prog/TruFor/CocoGlide.zip", 123161479,
     "fc848643ad0a81010ba2a6ab8c055762782f54e4fb9e407210dbfe1c3b538af9", "CocoGlide (research evaluation only)"),
    ("data", "realistic-tampering-dataset.zip", "https://pkorus.pl/downloads/dataset-realistic-tampering/realistic-tampering-dataset.zip", 1697070982,
     None, "Korus Realistic Tampering (educational/research only); per-image hashes in reports/manifests/korus_png.jsonl"),
    ("data", "dff_wiki.zip", "https://huggingface.co/datasets/OpenRL/DeepFakeFace/resolve/main/wiki.zip", 1766572515,
     "e83b55d6692d54e5aad38427531fa831bf30d1051ccc4267d4db90f03aaa7136", "DeepFakeFace real (IMDB-WIKI)"),
    ("data", "dff_insight.zip", "https://huggingface.co/datasets/OpenRL/DeepFakeFace/resolve/main/insight.zip", 1081375437,
     "3265a6f1eed49c55068c88929608d31c38db4588d942b9679b5f5ed1479ba5d7", "DeepFakeFace InsightFace swaps"),
    ("data", "dff_inpainting.zip", "https://huggingface.co/datasets/OpenRL/DeepFakeFace/resolve/main/inpainting.zip", 1082027665,
     "795b751f0967b2d940fbd9825e05c541666ec6f8c634acc7640044542897b7b2", "DeepFakeFace SD inpainting"),
    ("data", "dff_text2img.zip", "https://huggingface.co/datasets/OpenRL/DeepFakeFace/resolve/main/text2img.zip", 1134392707,
     "95e628873d1daea2b539a6bdf5036728a6879470ac37d6a487e091208db28f3b", "DeepFakeFace SD 1.5 text2img"),
]
NOTE_OPENFAKE = ("OpenFake (CC BY-NC 4.0): rows were read from core/test-00000-of-00013.parquet and core/test-00007-of-00013.parquet "
                 "(row groups listed in reports/manifests/openfake_row_groups.json) on huggingface.co/datasets/ComplexDataLab/OpenFake.")
NOTE_MEI = "ctrl_mei source photos: github.com/drewnoakes/metadata-extractor-images (list in reports/manifests/ctrl_mei_source_pool.json)."


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", type=Path, required=False)
    ap.add_argument("--only", choices=["weights", "data"], default=None)
    ap.add_argument("--list", action="store_true")
    a = ap.parse_args()
    rows = [s for s in SOURCES if a.only in (None, s[0])]
    if a.list or not a.dest:
        for kind, name, url, size, digest, note in rows:
            print(f"[{kind}] {name}  {size:,} B  sha256={digest or 'n/a'}\n    {url}\n    {note}")
        print(NOTE_OPENFAKE)
        print(NOTE_MEI)
        return 0
    a.dest.mkdir(parents=True, exist_ok=True)
    bad = 0
    for kind, name, url, size, digest, note in rows:
        target = a.dest / name
        if not (target.exists() and target.stat().st_size == size):
            print(f"downloading {name} ...", flush=True)
            tmp = target.with_suffix(target.suffix + ".part")
            req = urllib.request.Request(url, headers={"User-Agent": "ifd-fetch-sources/2.0"})
            with urllib.request.urlopen(req, timeout=120) as r, tmp.open("wb") as f:
                shutil.copyfileobj(r, f, 1 << 22)
            tmp.replace(target)
        ok_size = target.stat().st_size == size
        ok_hash = digest is None or sha256(target) == digest
        print(f"{'OK ' if ok_size and ok_hash else 'BAD'} {name} size={'ok' if ok_size else 'WRONG'} sha256={'ok' if ok_hash else 'WRONG'}")
        bad += not (ok_size and ok_hash)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
