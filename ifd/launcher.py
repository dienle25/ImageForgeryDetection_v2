"""Entry point used by RUN_APP.bat / run_app.sh: check the runtime, assemble shipped model parts, start the web UI.

Never downloads anything and never sends images anywhere; the web UI binds to 127.0.0.1 only.
Exit codes: 0 normal stop, 2 configuration/model problem, 3 missing/broken Python packages.
"""
from __future__ import annotations

import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
VC_REDIST = "https://aka.ms/vs/17/release/vc_redist.x64.exe"


def check_runtime() -> list[str]:
    problems = []
    if sys.version_info < (3, 10):
        problems.append(f"Cần Python ≥ 3.10 (đang dùng {platform.python_version()}).")
    for mod, pip_name in (("numpy", "numpy"), ("PIL", "pillow"), ("onnxruntime", "onnxruntime"),
                          ("cv2", "opencv-python-headless")):
        try:
            __import__(mod)
        except Exception as exc:  # ImportError, or OSError for a missing DLL on Windows
            msg = f"Không nạp được thư viện '{pip_name}': {type(exc).__name__}: {exc}"
            if os.name == "nt" and ("DLL" in str(exc) or isinstance(exc, OSError)):
                msg += ("\n    → Máy có thể thiếu Microsoft Visual C++ Redistributable 2015-2022 (x64). "
                        f"Cài từ trang chính thức của Microsoft: {VC_REDIST} rồi chạy lại.")
            problems.append(msg)
    return problems


def main(argv=None) -> int:
    os.environ.setdefault("OPENCV_LOG_LEVEL", "ERROR")
    print("=" * 72)
    print(" KIỂM TRA ẢNH CHỈNH SỬA / ẢNH SINH AI / DEEPFAKE — chạy CPU tại máy")
    print(" Ảnh KHÔNG được gửi ra Internet. Đóng cửa sổ này để dừng ứng dụng.")
    print("=" * 72)
    print(f"Python {platform.python_version()} · {platform.platform()} · thư mục: {ROOT}")
    problems = check_runtime()
    if problems:
        print("\nKHÔNG KHỞI ĐỘNG ĐƯỢC — thiếu/lỗi thư viện:", file=sys.stderr)
        for p in problems:
            print("  - " + p, file=sys.stderr)
        return 3
    from .errors import ForgeryError
    from .model_parts import assemble_all
    try:
        assemble_all(ROOT / "models")
    except (ForgeryError, OSError, ValueError) as exc:
        print(f"\nLỗi mô hình: {exc}", file=sys.stderr)
        return 2
    from .webapp.server import main as server_main
    return server_main(list(argv) if argv is not None else sys.argv[1:])


if __name__ == "__main__":
    raise SystemExit(main())
