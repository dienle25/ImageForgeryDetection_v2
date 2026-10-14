"""PSCC-Net inference on ONNX Runtime with explicit, configurable preprocessing policies."""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
from PIL import Image

_RESAMPLE = {"bilinear": Image.Resampling.BILINEAR, "bicubic": Image.Resampling.BICUBIC,
             "lanczos": Image.Resampling.LANCZOS, "nearest": Image.Resampling.NEAREST,
             "area": Image.Resampling.BOX}


def interp_ac(x: np.ndarray, size) -> np.ndarray:
    """torch.nn.functional.interpolate(bilinear, align_corners=True) for (C,H,W) float arrays."""
    c, ih, iw = x.shape
    oh, ow = size

    def coords(n_in, n_out):
        src = np.zeros(1) if n_out == 1 else np.arange(n_out, dtype=np.float64) * ((n_in - 1) / (n_out - 1))
        i0 = np.clip(np.floor(src).astype(np.int64), 0, n_in - 1)
        i1 = np.minimum(i0 + 1, n_in - 1)
        return i0, i1, (src - i0).astype(np.float32)

    y0, y1, fy = coords(ih, oh)
    x0, x1, fx = coords(iw, ow)
    top = x[:, y0, :] * (1 - fy)[None, :, None] + x[:, y1, :] * fy[None, :, None]
    return (top[:, :, x0] * (1 - fx)[None, None, :] + top[:, :, x1] * fx[None, None, :]).astype(np.float32)


@dataclass
class RawOutput:
    score: float
    logits: list
    map256: np.ndarray        # (256,256) head output for the model input


class PSCCSession:
    def __init__(self, onnx_path: str, threads: int | None = None):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        if threads:
            opts.intra_op_num_threads = int(threads)
            opts.inter_op_num_threads = 1
        self.sess = ort.InferenceSession(onnx_path, sess_options=opts, providers=["CPUExecutionProvider"])

    def run(self, x_chw01: np.ndarray) -> RawOutput:
        m, lg = self.sess.run(None, {"image": np.ascontiguousarray(x_chw01[None], dtype=np.float32)})
        lg = lg[0].astype(np.float64)
        e = np.exp(lg - lg.max())
        return RawOutput(float(e[1] / e.sum()), lg.tolist(), m[0, 0])


def to_input(rgb_u8: np.ndarray) -> np.ndarray:
    return rgb_u8.astype(np.float32).transpose(2, 0, 1) / 255.0


def baseline_policy(sess: PSCCSession, rgb: np.ndarray, max_side: int = 512):
    """Exactly the audited wrapper: bounded aspect-preserving bilinear downscale, edge pad to /8."""
    h, w = rgb.shape[:2]
    s = min(1.0, max_side / max(h, w))
    rh, rw = max(1, round(h * s)), max(1, round(w * s))
    work = np.asarray(Image.fromarray(rgb).resize((rw, rh), Image.Resampling.BILINEAR)) if (rh, rw) != (h, w) else rgb
    ph, pw = (-rh) % 8, (-rw) % 8
    work = np.pad(work, ((0, ph), (0, pw), (0, 0)), mode="edge")
    out = sess.run(to_input(work))
    full = interp_ac(out.map256[None], (rh + ph, rw + pw))[:, :rh, :rw]
    heat = interp_ac(full, (h, w))[0]
    return out.score, np.clip(heat, 0, 1), {"policy": "baseline", "max_side": max_side, "model_input_hw": [rh + ph, rw + pw], "logits": out.logits}


def resize_policy(sess: PSCCSession, rgb: np.ndarray, size: int = 256, resample: str = "bicubic", keep_aspect: bool = False):
    """Whole image resized to the training resolution (training loader: PIL bicubic to 256x256)."""
    h, w = rgb.shape[:2]
    if keep_aspect:
        s = size / max(h, w)
        th, tw = max(8, int(round(h * s / 8)) * 8), max(8, int(round(w * s / 8)) * 8)
    else:
        th, tw = size, size
    work = np.asarray(Image.fromarray(rgb).resize((tw, th), _RESAMPLE[resample]))
    out = sess.run(to_input(work))
    full = interp_ac(out.map256[None], (th, tw))
    heat = interp_ac(full, (h, w))[0]
    return out.score, np.clip(heat, 0, 1), {"policy": "resize", "size": size, "resample": resample,
                                           "keep_aspect": keep_aspect, "model_input_hw": [th, tw], "logits": out.logits}
