"""Validated image decoding.  All coordinates refer to the EXIF-oriented RGB image as displayed."""
from __future__ import annotations

import hashlib
import io
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

from .errors import ForgeryError

ALLOWED_FORMATS = ("JPEG", "PNG")
ALLOWED_MODES = ("RGB", "RGBA", "L", "LA", "P", "CMYK", "I;16", "I;16B", "I;16L", "I", "1")

# IJG standard luminance quantization table (quality 50), natural order.
_STD_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], dtype=np.float64)


@dataclass
class LoadedImage:
    rgb: np.ndarray
    source_name: str
    sha256: str
    metadata: dict = field(default_factory=dict)


def estimate_jpeg_quality(image: Image.Image) -> int | None:
    """Approximate IJG quality from the luminance quantization table (informational only)."""
    try:
        tables = getattr(image, "quantization", None)
        if not tables or 0 not in tables:
            return None
        q = np.asarray(tables[0], dtype=np.float64)
        if q.size != 64:
            return None
        best, best_err = None, None
        for quality in range(1, 101):
            scale = 5000 / quality if quality < 50 else 200 - 2 * quality
            ref = np.clip(np.floor((_STD_LUMA * scale + 50) / 100), 1, 255)
            # Pillow returns tables in zigzag or natural order depending on version; compare sorted values.
            err = float(np.abs(np.sort(ref) - np.sort(q)).mean())
            if best_err is None or err < best_err:
                best, best_err = quality, err
        return int(best) if best_err is not None and best_err < 2.0 else None
    except Exception:  # metadata only; never fail decoding because of it
        return None


def _exif_summary(image: Image.Image) -> dict:
    try:
        exif = image.getexif()
    except Exception:
        return {}
    names = {271: "make", 272: "model", 305: "software", 306: "datetime", 274: "orientation"}
    out = {}
    for tag, name in names.items():
        value = exif.get(tag)
        if value is not None:
            out[name] = str(value).strip("\x00 ")[:120]
    return out


def read_image(source: str | Path | bytes, *, max_file_mb: float, max_pixels: int, min_side: int,
               name: str | None = None) -> LoadedImage:
    limit = int(max_file_mb * 1024 * 1024)
    try:
        if isinstance(source, (bytes, bytearray)):
            data = bytes(source)
            source_name = Path(name or "upload").name
        else:
            path = Path(source)
            if not path.is_file():
                raise ForgeryError(f"Không tìm thấy file ảnh: {path}")
            if path.stat().st_size > limit:
                raise ForgeryError(f"Ảnh vượt giới hạn {max_file_mb:g} MiB.")
            data = path.read_bytes()
            source_name = path.name
        if not data:
            raise ForgeryError("File ảnh trống.")
        if len(data) > limit:
            raise ForgeryError(f"Ảnh vượt giới hạn {max_file_mb:g} MiB.")
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(io.BytesIO(data)) as probe:
                fmt = probe.format
                if fmt not in ALLOWED_FORMATS:
                    raise ForgeryError("Chỉ hỗ trợ nội dung JPEG hoặc PNG; đổi đuôi file không đổi định dạng.")
                width, height = probe.size
                if width * height > max_pixels:
                    raise ForgeryError(f"Ảnh vượt giới hạn {max_pixels:,} pixel.".replace(",", "."))
                if min(width, height) < min_side:
                    raise ForgeryError(f"Cạnh nhỏ nhất của ảnh phải ít nhất {min_side} pixel.")
                if getattr(probe, "n_frames", 1) != 1:
                    raise ForgeryError("Không hỗ trợ ảnh PNG động hoặc ảnh nhiều frame.")
                mode = probe.mode
                if mode not in ALLOWED_MODES:
                    raise ForgeryError(f"Không hỗ trợ chế độ màu {mode}.")
                probe.verify()
            with Image.open(io.BytesIO(data)) as image:
                image.load()
                jpeg_quality = estimate_jpeg_quality(image) if fmt == "JPEG" else None
                exif = _exif_summary(image)
                icc = bool(image.info.get("icc_profile"))
                progressive = bool(image.info.get("progressive") or image.info.get("progression"))
                orientation = image.getexif().get(274, 1) if hasattr(image, "getexif") else 1
                image = ImageOps.exif_transpose(image)
                notes = []
                if image.mode in ("I;16", "I;16B", "I;16L", "I"):
                    arr = np.asarray(image, dtype=np.float64)
                    peak = 65535.0 if arr.max() > 255 else 255.0
                    image = Image.fromarray(np.clip(np.round(arr / peak * 255), 0, 255).astype(np.uint8), "L")
                    notes.append("Ảnh 16-bit/xám đã được chuyển về 8-bit để đưa vào mô hình.")
                has_alpha = image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info)
                if has_alpha:
                    rgba = image.convert("RGBA")
                    background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
                    image = Image.alpha_composite(background, rgba).convert("RGB")
                    notes.append("Ảnh có kênh trong suốt (alpha) đã được ghép lên nền trắng trước khi phân tích; "
                                 "vùng trong suốt không mang dấu vết ảnh gốc.")
                else:
                    image = image.convert("RGB")
                rgb = np.array(image, dtype=np.uint8)
        if orientation not in (None, 1):
            notes.append("Đã xoay ảnh theo EXIF; mọi tọa độ kết quả theo ảnh đang hiển thị.")
        if mode == "CMYK":
            notes.append("Ảnh CMYK đã chuyển sang RGB bằng Pillow (không quản lý màu ICC).")
        if mode in ("L", "LA", "1"):
            notes.append("Ảnh xám được nhân thành 3 kênh RGB.")
        digest = hashlib.sha256(data).hexdigest()
        return LoadedImage(rgb, source_name, digest, {
            "format": fmt, "bytes": len(data), "sha256": digest,
            "encoded_size_wh": [width, height], "display_size_wh": [int(rgb.shape[1]), int(rgb.shape[0])],
            "original_mode": mode, "exif_orientation": orientation, "has_icc_profile": icc,
            "jpeg_quality_estimate": jpeg_quality, "jpeg_progressive": progressive if fmt == "JPEG" else None,
            "exif": exif, "coordinate_space": "EXIF-oriented display RGB",
            "metadata_note": "EXIF/ICC chỉ mang tính tham khảo, dễ bị sửa; không dùng làm bằng chứng thật/giả.",
            "warnings": notes,
        })
    except ForgeryError:
        raise
    except UnidentifiedImageError as exc:
        raise ForgeryError("Không đọc được ảnh JPEG/PNG hợp lệ: file hỏng hoặc không phải ảnh.") from exc
    except (Image.DecompressionBombError, Image.DecompressionBombWarning) as exc:
        raise ForgeryError("Ảnh có số điểm ảnh bất thường (nghi “decompression bomb”); từ chối xử lý.") from exc
    except (OSError, ValueError, SyntaxError) as exc:
        detail = str(exc).split(" at 0x")[0][:120]          # never show Python object addresses to users
        raise ForgeryError(f"Không đọc được ảnh JPEG/PNG hợp lệ (file hỏng hoặc bị cắt cụt): {detail}") from exc


def decoded_pixel_hash(rgb: np.ndarray) -> str:
    h = hashlib.sha256()
    h.update(np.asarray(rgb.shape, dtype="<i8").tobytes())
    h.update(np.ascontiguousarray(rgb).tobytes())
    return h.hexdigest()
