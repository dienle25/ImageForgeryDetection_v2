"""Research adapter: Mesorch (Zhu et al., AAAI 2025; MIT) in PyTorch, CPU, eval mode.

Official test pipeline: albumentations Resize(512,512, INTER_LINEAR) + ImageNet Normalize; output sigmoid map 512x512.
The model has NO image-level head (pred_label=None); any image score derived from the map must be validated.
"""
import os
import sys, types
from pathlib import Path
import numpy as np
import cv2
import torch

UP = Path(os.environ.get("IFD_UPSTREAM", Path(__file__).resolve().parents[3] / "upstream")) / "Mesorch"


def _stub_imdlbenco():
    if "IMDLBenCo" in sys.modules:
        return
    reg = types.ModuleType("IMDLBenCo.registry")
    class _Reg:
        def register_module(self, *a, **k):
            def deco(cls):
                return cls
            return deco
    reg.MODELS = _Reg()
    pkg = types.ModuleType("IMDLBenCo")
    pkg.registry = reg
    sys.modules["IMDLBenCo"] = pkg
    sys.modules["IMDLBenCo.registry"] = reg


class MesorchRunner:
    def __init__(self, ckpt: str, variant: str = "mesorch", threads: int = 2):
        _stub_imdlbenco()
        sys.path.insert(0, str(UP))
        torch.set_num_threads(threads)
        if variant == "mesorch":
            from mesorch import MesorchFull as M
        else:
            import mesorch_p
            M = [getattr(mesorch_p, n) for n in dir(mesorch_p) if n.lower().startswith("mesorch")][0]
        self.model = M(seg_pretrain_path=None, conv_pretrain=False)
        sd = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = sd.get("model", sd)
        missing, unexpected = self.model.load_state_dict(sd, strict=True)
        self.load_report = {"missing": list(missing), "unexpected": list(unexpected)}
        self.model.eval()
        self.mean = np.array([0.485, 0.456, 0.406], np.float32)
        self.std = np.array([0.229, 0.224, 0.225], np.float32)

    @torch.inference_mode()
    def predict(self, rgb: np.ndarray):
        h, w = rgb.shape[:2]
        x = cv2.resize(rgb, (512, 512), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
        x = (x - self.mean) / self.std
        t = torch.from_numpy(x.transpose(2, 0, 1))[None]
        out = self.model(t, torch.zeros(1, 1, 512, 512))
        m = out["pred_mask"][0, 0].numpy()
        heat = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        return np.clip(heat, 0, 1).astype(np.float32), m
