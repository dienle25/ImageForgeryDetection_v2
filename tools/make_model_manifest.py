"""(Re)write models/manifest.json: size + SHA-256 of every shipped model plus provenance and licence notes.

    python tools/make_model_manifest.py

The app refuses to load a model whose size or SHA-256 differs from this manifest (ifd/model_store.py).
Licence notes are facts about the upstream sources, NOT legal advice; see LICENSES_VI.md.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INFO = {
    "mesorch.onnx": {
        "task": "local_manipulation",
        "source": "https://github.com/scu-zjz/Mesorch (README checkpoint 'mesorch-98.pth', "
                  "sha256 6d8fcd7ce7616d819bec6a9ed461b27187101e67247f8b2d2483fdc1f25f685a)",
        "conversion": "tools/export_mesorch_onnx.py (torch 2.6 TorchScript exporter, opset 17, fp32); "
                      "max |diff| vs PyTorch 8.7e-7 on probe input",
        "license_code": "MIT (Copyright (c) 2024 K-Squad)",
        "license_weights": "no separate licence stated for the checkpoint; trained on CASIA v2, Fantastic Reality, "
                           "IMD2020, tampCOCO (research-oriented dataset terms) - commercial use needs legal review",
    },
    "commfor_vits16_384.onnx": {
        "task": "ai_generated",
        "source": "https://huggingface.co/OwensLab/commfor-model-384 (revision 6076002b)",
        "conversion": "tools/export_commfor_onnx.py (timm vit_small_patch16_384 + Linear(384,1), opset 17, fp32) from model.safetensors sha256 b89f36275f3bf5e2b040eee36597a8f19db051bff9a473a9cf7b2466284fb387; max |logit diff| vs PyTorch 5.8e-6",
        "license_code": "MIT (Community Forensics, Park & Owens, CVPR 2025)",
        "license_weights": "MIT per model card; training images come from many generators with their own terms - "
                           "commercial use needs legal review",
    },
    "face_clip_vitl14.onnx": {
        "task": "face_deepfake",
        "source": "https://huggingface.co/yermandy/deepfake-detection (model.torchscript, "
                  "sha256 55c54d3f4d301ce07f0d08126fc4202572d20ea6952b039bc7a949a43714157d)",
        "conversion": "tools/export_face_clip_onnx.py: plain-PyTorch rebuild verified bit-identical to the TorchScript "
                      "in bf16, exported with bf16 weights + Cast->float32 compute (ONNX vs torch fp32 |diff| 1.9e-6)",
        "license_code": "MIT (Copyright (c) 2025 Andy / yermandy)",
        "license_weights": "MIT per repository; base CLIP ViT-L/14 (OpenAI, MIT); fine-tuned on FaceForensics++ "
                           "(research-only terms) - commercial use needs legal review",
    },
    "face_detection_yunet_2023mar.onnx": {
        "task": "face_detection (support for face_deepfake)",
        "source": "https://github.com/opencv/opencv_zoo/tree/main/models/face_detection_yunet",
        "conversion": "none (original file)",
        "license_code": "MIT (OpenCV Zoo YuNet)",
        "license_weights": "MIT",
    },
    "pscc_net.onnx": {
        "task": "local_manipulation (v1 baseline, disabled by default)",
        "source": "https://github.com/proteus1991/PSCC-Net @ 53e5ff77d8dc5feddda060cd085f9b765761f816 "
                  "(checkpoint/HRNet, NLCDetection, DetectionHead)",
        "conversion": "tools/pscc_onnx.py (no torch; BatchNorm folded; deterministic) from weights_source/*.pth",
        "license_code": "MIT (PSCC-Net repository)",
        "license_weights": "no separate weights licence found",
    },
}


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main():
    files = {}
    for name, info in INFO.items():
        p = ROOT / "models" / name
        if not p.exists():
            raise SystemExit(f"missing model file: {p}")
        files[name] = {"size_bytes": p.stat().st_size, "sha256": sha256(p), **info}
    out = {"schema_version": 1, "files": files}
    (ROOT / "models" / "manifest.json").write_text(json.dumps(out, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    for k, v in files.items():
        print(f"{k:38s} {v['size_bytes']:>11d} {v['sha256']}")


if __name__ == "__main__":
    main()
