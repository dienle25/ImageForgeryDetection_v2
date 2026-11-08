"""Cảnh báo "Độ tin cậy thấp" theo chất lượng ảnh đầu vào — chỉ để hiển thị, KHÔNG đổi điểm, ngưỡng hay kết luận.

Dùng thông tin read_image() đã có (kích thước hiển thị, định dạng, EXIF, kênh alpha) và tên file.  Bật cảnh báo khi
có ít nhất một điều kiện:
  * ảnh nhỏ hơn 0,5 MP (500 000 điểm ảnh);
  * dấu hiệu ảnh chụp màn hình: PNG có kênh alpha nhưng không có pixel trong suốt nào, HOẶC tên file kiểu
    "Screenshot…", "Ảnh chụp màn hình…", "Ảnh màn hình…";
  * JPEG không có EXIF và nhỏ hơn 2 MP (thường là ảnh đã qua ứng dụng nhắn tin, bị nén lại).
Riêng "không có EXIF" thì KHÔNG cảnh báo: ảnh tải từ Zalo, Facebook… đều mất EXIF.
"Có EXIF" = có ít nhất một trường EXIF ngoài các trường kỹ thuật mà bộ mã hóa nào cũng có thể ghi (hướng xoay,
độ phân giải, vị trí YCbCr).
"""
from __future__ import annotations

import math
import re
import unicodedata

RULE_VERSION = 1
MIN_MEGAPIXELS = 0.5
MESSAGING_MAX_MEGAPIXELS = 2.0
HEADLINE_VI = "Độ tin cậy thấp: ảnh nhỏ hoặc đã qua nén lại"
NOTE_VI = "Chỉ là cảnh báo: không đổi điểm, ngưỡng hay kết luận của mô hình."
# IFD0 tags written by almost any encoder; they say nothing about the camera origin of the picture
TECHNICAL_EXIF_TAGS = frozenset({274, 282, 283, 296, 531})   # Orientation, X/YResolution, ResolutionUnit, YCbCrPositioning
SCREENSHOT_NAME_PARTS = ("screenshot", "screen shot", "chup man hinh", "anh man hinh")
REASONS_VI = {
    "small_image": "ảnh nhỏ hơn 0,5 MP",
    "png_alpha_opaque": "PNG có kênh alpha nhưng không có pixel trong suốt",
    "screenshot_name": "tên file giống ảnh chụp màn hình",
    "jpeg_no_exif_small": "JPEG không EXIF, nhỏ hơn 2 MP",
}


def fold(text: str) -> str:
    """Lower case, no Vietnamese diacritics, '_' '-' '.' as spaces: 'Ảnh_chụp-màn hình' -> 'anh chup man hinh'."""
    s = unicodedata.normalize("NFD", str(text)).replace("đ", "d").replace("Đ", "D")
    s = "".join(c for c in s if unicodedata.category(c) != "Mn").lower()
    for ch in "_-.":
        s = s.replace(ch, " ")
    return " ".join(s.split())


def looks_like_screenshot_name(name: str | None) -> bool:
    """'Screenshot…', 'DC_Screenshot…', 'Ảnh chụp màn hình…', 'Ảnh màn hình…' (macOS tiếng Việt).  A pattern must start
    a word, so 'thanh_man_hinh.png' is not taken for 'ảnh màn hình'."""
    if not name:
        return False
    base = name.rsplit("/", 1)[-1].rsplit("\\", 1)[-1]
    folded = fold(base)
    return any(re.search(r"(?:^| )" + re.escape(part), folded) for part in SCREENSHOT_NAME_PARTS)


def mp_text(mp: float) -> str:
    """Megapixels rounded DOWN, Vietnamese decimal comma: 0.3456 -> '0,34'; never shows a value above the real one."""
    digits = 3 if mp < 0.1 else 2
    q = math.floor(mp * 10 ** digits) / 10 ** digits
    return f"{q:.{digits}f}".replace(".", ",")


def has_exif(meta: dict) -> bool:
    tags = meta.get("exif_tags")
    if tags is None:                         # older metadata without the tag list: use the extracted fields
        return any(k != "orientation" for k in (meta.get("exif") or {}))
    return bool(set(tags) - TECHNICAL_EXIF_TAGS)


def assess(meta: dict, source_name: str | None = None) -> dict:
    """meta = LoadedImage.metadata from ifd.image_io.read_image.  Returns the 'input_quality' block of the report."""
    w, h = meta["display_size_wh"]
    mp = (int(w) * int(h)) / 1e6
    fmt = meta.get("format")
    alpha = meta.get("alpha") or {}
    reasons = []
    if mp < MIN_MEGAPIXELS:
        reasons.append({"code": "small_image", "text_vi": f"{mp_text(mp)} MP, nhỏ hơn 0,5 MP"})
    if fmt == "PNG" and alpha.get("present") and alpha.get("min") == 255:
        reasons.append({"code": "png_alpha_opaque",
                        "text_vi": "PNG có kênh alpha nhưng không có pixel trong suốt nào (dấu hiệu ảnh chụp màn hình)"})
    if looks_like_screenshot_name(source_name):
        shown = str(source_name)
        shown = shown if len(shown) <= 60 else shown[:57] + "…"
        reasons.append({"code": "screenshot_name", "text_vi": f"tên file “{shown}” giống ảnh chụp màn hình"})
    if fmt == "JPEG" and not has_exif(meta) and mp < MESSAGING_MAX_MEGAPIXELS:
        reasons.append({"code": "jpeg_no_exif_small",
                        "text_vi": f"JPEG không có EXIF và {mp_text(mp)} MP, nhỏ hơn 2 MP (thường là ảnh đã qua ứng dụng "
                                   "nhắn tin, bị nén lại)"})
    low = bool(reasons)
    return {"rule_version": RULE_VERSION, "low_confidence": low,
            "headline_vi": HEADLINE_VI if low else None,
            "text_vi": (HEADLINE_VI + " — " + "; ".join(r["text_vi"] for r in reasons)) if low else None,
            "reasons": reasons, "megapixels": round(mp, 4), "changes_decision": False, "note_vi": NOTE_VI}


def compact(block: dict | None) -> dict | None:
    """Short form kept per image by the validation kit (cache line, du_doan.csv)."""
    if not block:
        return None
    return {"v": block.get("rule_version", RULE_VERSION), "low": bool(block.get("low_confidence")),
            "codes": [r["code"] for r in block.get("reasons", [])],
            "text_vi": "; ".join(r["text_vi"] for r in block.get("reasons", []))}
