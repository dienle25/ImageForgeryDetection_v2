"""Model files: SHA-256 verification against models/manifest.json before loading."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .errors import ForgeryError


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def verify_model(path: Path, manifest_path: Path) -> dict:
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ForgeryError(f"Không đọc được danh mục mô hình {manifest_path.name}: {exc}") from exc
    entry = manifest.get("files", {}).get(path.name)
    if entry is None:
        raise ForgeryError(f"Mô hình {path.name} không có trong {manifest_path.name}; không nạp file lạ.")
    if not path.is_file():
        raise ForgeryError(f"Thiếu file mô hình {path.name}. Chạy: python scripts/fetch_models.py (xem README).")
    if path.stat().st_size != entry["size_bytes"]:
        raise ForgeryError(f"File mô hình {path.name} sai dung lượng (hỏng hoặc tải chưa đủ).")
    digest = sha256_file(path)
    if digest != entry["sha256"]:
        raise ForgeryError(f"File mô hình {path.name} sai SHA-256; không sử dụng file này.")
    return {"file": path.name, "sha256": digest, **{k: v for k, v in entry.items() if k not in ("sha256",)}}
