"""Large model files may be shipped as numbered parts (name.onnx.part001, ...) to fit transfer limits.

`assemble_all()` rebuilds every model listed in models/manifest.json that is missing but has parts, verifies
size + SHA-256 against the manifest, and only then removes the parts.  Nothing is downloaded.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

from .errors import ForgeryError

PART_SUFFIX = ".part"


def parts_of(target: Path) -> list[Path]:
    parts = sorted(target.parent.glob(target.name + PART_SUFFIX + "[0-9][0-9][0-9]"))
    return parts


def assemble(target: Path, entry: dict, log=print) -> bool:
    """Return True if the file was assembled now; False if it already existed or has no parts."""
    if target.is_file():
        return False
    parts = parts_of(target)
    if not parts:
        return False
    expected = [target.name + PART_SUFFIX + f"{i:03d}" for i in range(1, len(parts) + 1)]
    if [p.name for p in parts] != expected:
        raise ForgeryError(f"Thiếu phần của mô hình {target.name}: cần {expected[0]}…{expected[-1]} liên tục.")
    total = sum(p.stat().st_size for p in parts)
    if total != entry["size_bytes"]:
        raise ForgeryError(f"Các phần của {target.name} có tổng {total} byte, cần {entry['size_bytes']} byte "
                           "(tải/giải nén chưa đủ).")
    log(f"Đang ghép mô hình {target.name} từ {len(parts)} phần…")
    tmp = target.with_name(target.name + ".assembling")
    h = hashlib.sha256()
    with tmp.open("wb") as out:
        for p in parts:
            with p.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 22), b""):
                    h.update(chunk)
                    out.write(chunk)
    if h.hexdigest() != entry["sha256"]:
        tmp.unlink(missing_ok=True)
        raise ForgeryError(f"Mô hình {target.name} ghép xong nhưng sai SHA-256; giữ nguyên các phần, không dùng file.")
    os.replace(tmp, target)
    for p in parts:
        try:
            p.unlink()
        except OSError:
            pass
    log(f"  OK {target.name} (SHA-256 khớp danh mục).")
    return True


def assemble_all(models_dir: Path, log=print) -> list[str]:
    manifest = json.loads((models_dir / "manifest.json").read_text(encoding="utf-8"))
    done = []
    for name, entry in manifest.get("files", {}).items():
        if assemble(models_dir / name, entry, log):
            done.append(name)
    return done


def split_file(path: Path, part_bytes: int) -> list[Path]:
    """Build-time helper: split `path` into numbered parts next to it (the original is kept)."""
    out = []
    with path.open("rb") as f:
        i = 1
        while True:
            chunk = f.read(part_bytes)
            if not chunk:
                break
            p = path.with_name(path.name + PART_SUFFIX + f"{i:03d}")
            p.write_bytes(chunk)
            out.append(p)
            i += 1
    return out
