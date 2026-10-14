"""Resampling helpers (NumPy only)."""
from __future__ import annotations

import numpy as np
from PIL import Image

PIL_FILTERS = {"bilinear": Image.Resampling.BILINEAR, "bicubic": Image.Resampling.BICUBIC,
               "lanczos": Image.Resampling.LANCZOS, "nearest": Image.Resampling.NEAREST,
               "area": Image.Resampling.BOX}


def interp_align_corners(x: np.ndarray, size) -> np.ndarray:
    """Equivalent of torch F.interpolate(mode='bilinear', align_corners=True) for (C,H,W) arrays."""
    c, ih, iw = x.shape
    oh, ow = int(size[0]), int(size[1])

    def coords(n_in, n_out):
        src = np.zeros(1) if n_out == 1 else np.arange(n_out, dtype=np.float64) * ((n_in - 1) / (n_out - 1))
        i0 = np.clip(np.floor(src).astype(np.int64), 0, n_in - 1)
        i1 = np.minimum(i0 + 1, n_in - 1)
        return i0, i1, (src - i0).astype(np.float32)

    y0, y1, fy = coords(ih, oh)
    x0, x1, fx = coords(iw, ow)
    top = x[:, y0, :] * (1 - fy)[None, :, None] + x[:, y1, :] * fy[None, :, None]
    return (top[:, :, x0] * (1 - fx)[None, None, :] + top[:, :, x1] * fx[None, None, :]).astype(np.float32)


def resize_rgb(rgb: np.ndarray, size_hw, method: str) -> np.ndarray:
    h, w = int(size_hw[0]), int(size_hw[1])
    if (h, w) == rgb.shape[:2]:
        return rgb
    return np.asarray(Image.fromarray(rgb).resize((w, h), PIL_FILTERS[method]))
