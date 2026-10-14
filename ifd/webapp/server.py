"""Local web UI: Python standard-library HTTP server bound to 127.0.0.1 + static HTML/JS.

* Images never leave the machine: the browser posts bytes to this local process only.
* A random per-launch token is required on every API call (blocks other websites/processes).
* Host header is checked (DNS-rebinding protection); results are kept in RAM only (LRU).
"""
from __future__ import annotations

import argparse
import collections
import json
import logging
import mimetypes
import os
import secrets
import socket
import sys
import threading
import time
import traceback
import urllib.parse
import webbrowser
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


from .. import __version__
from ..analysis import Analyzer, report_files, zip_bytes
from ..config import DEFAULT_CONFIG, load_config
from ..detectors.base import TASKS
from ..errors import ForgeryError
from ..render import jpeg_preview

STATIC = Path(__file__).resolve().parent / "static"
log = logging.getLogger("ifd.webapp")
MAX_RESULTS = 8
MAX_RESULT_BYTES = 600 * 1024 * 1024     # 12 MP photos produce ~90 MB of full-resolution files per result


class State:
    """Shared app state.  Models load in the background so the page can show progress immediately."""

    def __init__(self, config_path: Path, analyzer=None):
        self.token = secrets.token_urlsafe(18)
        self.config_path = config_path
        self.config = None
        self.analyzer = analyzer
        self.error = None
        self.loading = False
        self.results = collections.OrderedDict()
        self.lock = threading.Lock()
        try:
            self.config = load_config(config_path)
        except ForgeryError as exc:
            self.error = str(exc)

    def load(self) -> None:
        if self.analyzer is not None or self.config is None:
            return
        self.loading = True
        try:
            self.analyzer = Analyzer(self.config)
        except ForgeryError as exc:
            self.error = str(exc)
        except MemoryError:
            self.error = "Không đủ RAM để nạp các mô hình (cần khoảng 3–4 GB trống). Đóng bớt ứng dụng khác (kể cả cửa sổ PHAN_TICH_THU_MUC đang chạy) rồi chạy lại."
        except Exception as exc:  # unexpected: keep the server up to show the error
            log.exception("Không khởi tạo được bộ phân tích")
            self.error = f"Lỗi khởi tạo không mong đợi: {type(exc).__name__}: {exc}"
        finally:
            self.loading = False

    def status(self) -> dict:
        data = {"app_version": __version__, "ready": self.analyzer is not None, "loading": self.loading,
                "error": self.error}
        if self.config is not None:
            dets = []
            from ..analysis import load_validation_summary
            summaries = load_validation_summary(self.config.base_dir)   # written after the final test
            for key, det in self.config.data["detectors"].items():
                if det.get("enabled"):
                    obj = self.analyzer.detectors.get(key) if self.analyzer else None
                    task = getattr(obj, "task", None) or det.get("task") or TYPE_TASK.get(det.get("type"), "local_manipulation")
                    dets.append({"key": key, "task": task, "task_vi": TASKS.get(task, task),
                                 "title": det.get("title_vi", key), "thresholds": det.get("thresholds"),
                                 "validation": (summaries.get(key) or {}).get("text_vi"),
                                 "validation_status": (summaries.get(key) or {}).get("status")})
            covered = {d["task"] for d in dets}
            if self.analyzer is not None:
                data["config_matches_evaluated"] = self.analyzer.config_matches_evaluated()
                data["evaluated_profile"] = self.analyzer.evaluated_profile()
            data.update(detectors=dets, max_file_mb=self.config.limits["max_file_mb"],
                        tasks_not_checked=[TASKS[t] for t in TASKS if t not in covered],
                        models=self.analyzer.model_info if self.analyzer else None)
        return data

    def store(self, payload: dict) -> str:
        """Keep at most MAX_RESULTS results and MAX_RESULT_BYTES in RAM (oldest dropped first; newest always kept)."""
        rid = secrets.token_hex(8)
        payload["_bytes"] = sum(len(v) for v in payload["files"].values()) + sum(len(v) for v in payload["previews"].values())
        with self.lock:
            self.results[rid] = payload
            while len(self.results) > MAX_RESULTS or (
                    len(self.results) > 1 and sum(p["_bytes"] for p in self.results.values()) > MAX_RESULT_BYTES):
                self.results.popitem(last=False)
        return rid


TYPE_TASK = {"pscc_onnx": "local_manipulation", "mesorch_onnx": "local_manipulation",
             "commfor_onnx": "ai_generated", "face_clip_onnx": "face_deepfake"}


def make_handler(state: State, allowed_hosts: set[str]):
    class Handler(BaseHTTPRequestHandler):
        server_version = "IFD/" + __version__
        sys_version = ""

        def log_message(self, fmt, *args):  # keep the console readable
            log.info("%s - %s", self.address_string(), fmt % args)

        # ---------- helpers ----------
        def _host_ok(self) -> bool:
            return self.headers.get("Host", "") in allowed_hosts

        def _token_ok(self, query: dict) -> bool:
            supplied = self.headers.get("X-IFD-Token") or (query.get("t", [""])[0])
            return bool(supplied) and secrets.compare_digest(supplied, state.token)

        def _send(self, code: int, body: bytes, ctype: str, extra: dict | None = None):
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Referrer-Policy", "no-referrer")
            self.send_header("Content-Security-Policy",
                             "default-src 'self'; img-src 'self' blob: data:; style-src 'self'; script-src 'self'; "
                             "connect-src 'self'; frame-ancestors 'none'")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()
            if self.command != "HEAD":
                self.wfile.write(body)

        def _json(self, code: int, data: dict):
            self._send(code, json.dumps(data, ensure_ascii=False).encode("utf-8"), "application/json; charset=utf-8")

        # ---------- routes ----------
        def do_GET(self):
            if not self._host_ok():
                return self._send(HTTPStatus.FORBIDDEN, b"Forbidden host", "text/plain")
            url = urllib.parse.urlparse(self.path)
            query = urllib.parse.parse_qs(url.query)
            if url.path in ("/", "/index.html"):
                return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            if url.path.startswith("/static/"):
                name = url.path[len("/static/"):]
                target = (STATIC / name).resolve()
                if STATIC not in target.parents or not target.is_file():
                    return self._send(404, b"Not found", "text/plain")
                ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
                if ctype.startswith("text/") or ctype.endswith("javascript"):
                    ctype += "; charset=utf-8"
                return self._send(200, target.read_bytes(), ctype)
            if not self._token_ok(query):
                return self._json(HTTPStatus.FORBIDDEN, {"error": "Thiếu hoặc sai mã phiên. Mở lại đúng địa chỉ in trong cửa sổ chạy ứng dụng."})
            if url.path == "/api/status":
                return self._json(200, state.status())
            if url.path.startswith("/api/result/"):
                parts = url.path.split("/")
                if len(parts) != 5:
                    return self._send(404, b"Not found", "text/plain")
                rid, name = parts[3], parts[4]
                with state.lock:
                    payload = state.results.get(rid)
                if payload is None:
                    return self._json(404, {"error": "Kết quả đã hết hạn (chỉ giữ vài lần phân tích gần nhất trong RAM). Hãy phân tích lại."})
                if name == "results.zip":
                    body, ctype = zip_bytes(payload["files"]), "application/zip"
                elif name in payload["files"]:
                    body = payload["files"][name]
                    ctype = mimetypes.guess_type(name)[0] or "application/octet-stream"
                elif name in payload["previews"]:
                    body, ctype = payload["previews"][name], "image/jpeg"
                else:
                    return self._send(404, b"Not found", "text/plain")
                disp = {"Content-Disposition": "attachment; filename*=UTF-8''" + urllib.parse.quote(payload["stem"] + "_" + name)} \
                    if query.get("download") else {}
                return self._send(200, body, ctype, disp)
            return self._send(404, b"Not found", "text/plain")

        def do_POST(self):
            if not self._host_ok():
                return self._send(HTTPStatus.FORBIDDEN, b"Forbidden host", "text/plain")
            url = urllib.parse.urlparse(self.path)
            if not self._token_ok(urllib.parse.parse_qs(url.query)):
                return self._json(HTTPStatus.FORBIDDEN, {"error": "Thiếu hoặc sai mã phiên."})
            if url.path != "/api/analyze":
                return self._send(404, b"Not found", "text/plain")
            if state.analyzer is None:
                msg = state.error or ("Đang nạp mô hình, vui lòng chờ rồi bấm lại." if state.loading
                                      else "Bộ phân tích chưa sẵn sàng.")
                return self._json(503, {"error": msg})
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = -1
            limit = int(state.config.limits["max_file_mb"] * 1024 * 1024)
            if length <= 0:
                return self._json(400, {"error": "Chưa nhận được dữ liệu ảnh."})
            if length > limit:
                return self._json(413, {"error": f"Ảnh vượt giới hạn {state.config.limits['max_file_mb']:g} MiB."})
            body = self.rfile.read(length)
            name = urllib.parse.unquote(self.headers.get("X-File-Name", "upload"))[:200]
            client_req = self.headers.get("X-Request-Id", "")[:64]
            t0 = time.perf_counter()
            try:
                report = state.analyzer.analyze(body, name=name)
            except ForgeryError as exc:
                return self._json(422, {"error": str(exc), "request_id": client_req})
            except Exception as exc:
                log.error("Lỗi không mong đợi khi phân tích %s:\n%s", name, traceback.format_exc())
                return self._json(500, {"error": f"Lỗi không mong đợi ({type(exc).__name__}). Xem cửa sổ chạy ứng dụng để biết chi tiết.",
                                        "request_id": client_req})
            arrays: dict = {}
            files = report_files(report, state.config.overlay_alpha, arrays)   # renders heatmap/overlay ONCE
            previews = {"preview_original.jpg": jpeg_preview(report.image.rgb)}
            if "heatmap" in arrays:
                previews["preview_heatmap.jpg"] = jpeg_preview(arrays["heatmap"])
                previews["preview_overlay.jpg"] = jpeg_preview(arrays["overlay"])
            if "faces" in arrays:
                previews["preview_faces.jpg"] = jpeg_preview(arrays["faces"])
            del arrays
            stem = Path(name).stem or "image"
            rid = state.store({"files": files, "previews": previews, "stem": "".join(c if c.isalnum() or c in "-_" else "_" for c in stem)[:60]})
            data = json.loads(files["result.json"])
            return self._json(200, {"result_id": rid, "request_id": client_req, "report": data,
                                    "files": sorted(files), "previews": sorted(previews),
                                    "server_seconds": time.perf_counter() - t0})

        do_HEAD = do_GET

    return Handler


class LocalHTTPServer(ThreadingHTTPServer):
    """127.0.0.1-only server.  On Windows SO_REUSEADDR would let a second process silently bind a port that is
    already in use, so it is disabled there and SO_EXCLUSIVEADDRUSE is requested instead."""
    daemon_threads = True
    allow_reuse_address = os.name != "nt"

    def server_bind(self):
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def free_port(host: str) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind((host, 0))
        return s.getsockname()[1]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Giao diện web cục bộ (chỉ 127.0.0.1).")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--port", type=int, default=0, help="0 = tự chọn cổng trống")
    ap.add_argument("--no-browser", action="store_true")
    ap.add_argument("--print-url-file", type=Path, help="(kiểm thử) ghi URL vào file này")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    host = "127.0.0.1"
    if not 0 <= args.port <= 65535:
        print(f"Cổng không hợp lệ: {args.port} (0–65535).", file=sys.stderr)
        return 2
    state = State(args.config)
    if state.error:
        print(f"Lỗi cấu hình: {state.error}", file=sys.stderr)
        return 2
    port = args.port or free_port(host)
    # bind BEFORE loading models: a busy port fails fast with a clear message
    try:
        httpd = LocalHTTPServer((host, port), make_handler(state, {f"{host}:{port}", f"localhost:{port}"}))
    except OSError as exc:
        print(f"Không mở được cổng {port}: {exc}. Có thể cổng đang được dùng; chạy lại để tự chọn cổng khác "
              "(hoặc dùng --port 0).", file=sys.stderr)
        return 2
    url = f"http://{host}:{port}/?t={state.token}"
    print(f"Ứng dụng đang chạy tại: {url}\nGiữ cửa sổ này mở trong lúc dùng. Nhấn Ctrl+C hoặc đóng cửa sổ để dừng.",
          flush=True)
    if args.print_url_file:
        args.print_url_file.write_text(url, encoding="utf-8")
    loader = threading.Thread(target=_load_and_report, args=(state,), daemon=True)
    loader.start()
    if not args.no_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


def _load_and_report(state: "State") -> None:
    t0 = time.perf_counter()
    print("Đang nạp và kiểm tra mô hình (lần đầu có thể mất 30–120 giây)...", flush=True)
    state.load()
    if state.error:
        print(f"LỖI: {state.error}", file=sys.stderr, flush=True)
    else:
        print(f"Sẵn sàng sau {time.perf_counter() - t0:.0f} giây. Mở trình duyệt theo địa chỉ ở trên.", flush=True)


if __name__ == "__main__":
    raise SystemExit(main())
