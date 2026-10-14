"""Research adapter: SPAI — "Any-Resolution AI-Generated Image Detection by Spectral Learning" (CVPR 2025).
Code + weights: Apache-2.0 (github.com/mever-team/spai). ViT-B/16 masked-frequency backbone, applied to
non-overlapping 224x224 patches of the image at ORIGINAL resolution, patch features fused by cross-attention.

Preprocessing replicated from spai/data/data_finetune.py (inference branch, ORIGINAL_RESOLUTION=True,
REQUIRED_NORMALIZATION=positive_0_1): uint8 RGB -> float /255, PadIfNeeded(min 224, BORDER_REFLECT_101,
centered); no resize unless `max_side` is given (SmallestMaxSize semantics when `resize_to` is used upstream).
Output: logit of class "AI-generated"; score = sigmoid(logit).
"""
from __future__ import annotations

import os
import sys
import types
from pathlib import Path

import cv2
import numpy as np
import torch

SPAI_ROOT_DEFAULT = Path(os.environ.get("IFD_UPSTREAM", Path(__file__).resolve().parents[3] / "upstream")) / "spai"


def _import_spai(root: Path):
    # The OpenAI `clip` package is only needed for the CLIP backbone variant, which spai.yaml does not use.
    sys.modules.setdefault("clip", types.ModuleType("clip"))
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))
    from spai.config import get_config  # noqa
    from spai.models.build import build_cls_model  # noqa
    return get_config, build_cls_model


def preprocess(rgb: np.ndarray, img_size: int = 224, smallest_max_size: int | None = None) -> torch.Tensor:
    if smallest_max_size is not None:
        h, w = rgb.shape[:2]
        s = smallest_max_size / min(h, w)
        if s != 1.0:
            rgb = cv2.resize(rgb, (round(w * s), round(h * s)), interpolation=cv2.INTER_LINEAR)
    h, w = rgb.shape[:2]
    ph, pw = max(0, img_size - h), max(0, img_size - w)
    if ph or pw:
        rgb = cv2.copyMakeBorder(rgb, ph // 2, ph - ph // 2, pw // 2, pw - pw // 2, cv2.BORDER_REFLECT_101)
    x = torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1).float().div_(255.0)
    return x.unsqueeze(0)


class SpaiRunner:
    def __init__(self, ckpt: str, threads: int = 2, spai_root: Path = SPAI_ROOT_DEFAULT,
                 feature_batch: int = 8, smallest_max_size: int | None = None):
        torch.set_num_threads(threads)
        get_config, build_cls_model = _import_spai(Path(spai_root))
        cfg = get_config({"cfg": str(Path(spai_root) / "configs" / "spai.yaml")})
        self.cfg = cfg
        self.model = build_cls_model(cfg)
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = ck["model"] if "model" in ck else ck
        if any(k.startswith("encoder.") for k in sd):
            sd = {k.replace("encoder.", ""): v for k, v in sd.items() if k.startswith("encoder.")}
        res = self.model.load_state_dict(sd, strict=False)
        # Anything missing other than non-persistent buffers would silently randomize part of the network.
        self.missing = [k for k in res.missing_keys]
        self.unexpected = [k for k in res.unexpected_keys]
        if self.missing:
            raise RuntimeError(f"SPAI checkpoint missing keys: {self.missing[:10]} (n={len(self.missing)})")
        self.model.eval()
        self.feature_batch = feature_batch
        self.smallest_max_size = smallest_max_size
        self.img_size = int(cfg.DATA.IMG_SIZE)

    @torch.inference_mode()
    def predict(self, rgb: np.ndarray) -> dict:
        x = preprocess(rgb, self.img_size, self.smallest_max_size)
        logit = self.model([x], self.feature_batch)
        lg = float(logit.reshape(-1)[0])
        h, w = x.shape[-2:]
        n_patches = max(1, (h - 224) // 224 + 1) * max(1, (w - 224) // 224 + 1)
        return {"score": float(1.0 / (1.0 + np.exp(-lg))), "logit": lg,
                "input_hw": [int(h), int(w)], "patches": int(max(n_patches, 5 if n_patches < 4 else n_patches))}
