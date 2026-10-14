class ForgeryError(Exception):
    """An actionable error that is safe to show to the user (Vietnamese message)."""


_RAM_HINT = (" — Có thể do thiếu RAM: đóng các ứng dụng khác (kể cả cửa sổ RUN_APP hoặc PHAN_TICH_THU_MUC đang chạy) "
             "rồi thử lại; bộ mô hình cần khoảng 3–4 GB RAM trống.")


def model_load_error(what: str, path, exc: BaseException) -> ForgeryError:
    """Vietnamese error for a model that could not be loaded; adds a RAM hint when the cause looks like memory."""
    text = str(exc).strip() or type(exc).__name__
    low = text.lower()
    hint = _RAM_HINT if isinstance(exc, MemoryError) or "alloc" in low or "out of memory" in low else ""
    return ForgeryError(f"Không nạp được mô hình {what} ({path}): {text[:400]}{hint}")
