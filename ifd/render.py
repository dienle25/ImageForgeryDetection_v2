"""Visualisation with a FIXED [0,1] colour scale (never normalised per image)."""
from __future__ import annotations

import io

import numpy as np
from PIL import Image

_ANCHORS = np.array([[20, 35, 120], [0, 170, 225], [255, 230, 55], [210, 25, 35]], dtype=np.float32)


def colorize(heatmap: np.ndarray) -> np.ndarray:
    """Fixed colour scale.  Computed channel by channel with the same float32 operations as the original
    vectorised form (bit-identical output, ~5x faster on 12 MP maps: no (H, W, 3) gathers)."""
    scaled = np.clip(heatmap, 0, 1) * 3
    index = np.minimum(scaled.astype(np.int32), 2)
    weight = scaled - index
    inverse = 1 - weight
    out = np.empty(np.shape(heatmap) + (3,), dtype=np.uint8)
    for ch in range(3):
        lut = _ANCHORS[:, ch]
        out[..., ch] = np.round(lut[index] * inverse + lut[index + 1] * weight)
    return out


def overlay(rgb: np.ndarray, heatmap: np.ndarray, alpha: float, colored: np.ndarray | None = None) -> np.ndarray:
    """`colored` = colorize(heatmap) if the caller already has it (same result, computed once)."""
    opacity = (alpha * np.clip(heatmap, 0, 1))[..., None]
    col = colorize(heatmap) if colored is None else colored
    return np.round(rgb * (1 - opacity) + col * opacity).astype(np.uint8)


def binary_mask(heatmap: np.ndarray, threshold: float) -> np.ndarray:
    return (heatmap >= threshold).astype(np.uint8) * 255


_DECISION_RGB = {"suspicious": (210, 25, 35), "needs_review": (245, 160, 0), "no_clear_sign": (30, 150, 70)}


def draw_face_boxes(rgb: np.ndarray, faces: list[dict]) -> np.ndarray:
    """Rectangles (colour = per-face decision) and a face index tag; no text rendering dependency."""
    from PIL import ImageDraw
    im = Image.fromarray(rgb).copy()
    d = ImageDraw.Draw(im)
    lw = max(2, round(max(im.size) / 400))
    for i, f in enumerate(faces, 1):
        x, y, w, h = f["box_xywh"]
        col = _DECISION_RGB.get(f.get("decision"), (255, 255, 255))
        d.rectangle([x, y, x + w, y + h], outline=col, width=lw)
        tag = f"#{i} {f['score']:.2f}"
        tx, ty = x, max(0, y - 12 * lw)
        d.rectangle([tx, ty, tx + 7 * lw * len(tag) // 2 + 4, ty + 11 * lw], fill=col)
        d.text((tx + 2, ty + 1), tag, fill=(255, 255, 255))
    return np.asarray(im)


def png_bytes(array: np.ndarray) -> bytes:
    """Lossless PNG; zlib level 1 (pixels identical to any other level, ~30 % faster on 12 MP images)."""
    buf = io.BytesIO()
    Image.fromarray(array).save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def jpeg_preview(array: np.ndarray, max_side: int = 1600, quality: int = 88) -> bytes:
    im = Image.fromarray(array)
    if max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()
