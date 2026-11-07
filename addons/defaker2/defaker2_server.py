"""DefakerOne2 scoring server: an add-on for ImageForgeryDetection_v2, run in the conda env `defaker2`.

    python defaker2_server.py --repo D:\\deep_fake\\DefakerOne2 --model D:\\deep_fake\\models\\DefakerOne2\\release
                              [--device auto|gpu|cpu] [--port 8765]

* Loads the model ONCE and listens on 127.0.0.1 only (never on the network).
* POST /score  body = JPEG or PNG bytes (Content-Type image/png, image/jpeg or application/octet-stream)
               -> {"p_fake", "p_real", "device", "seconds", "mode", "answer", "top_logprobs", ...}
  GET  /health -> {"ready", "loading", "error", "device", "mode", ...}
* Inference = the README pipeline of DefakerOne2, run through ms-swift exactly like `swift infer`:
  prompt "<image>Is this image real or fake? Answer only Real or Fake", MIN_PIXELS=262144, MAX_PIXELS=2408448,
  --max_new_tokens 16 --enable_thinking False --logprobs True --top_logprobs 20.
  Score = logprob2score.logprobs_to_probs() imported from the DefakerOne2 repository (same code as the README).
* Device: GPU = fp16 + bitsandbytes 8-bit (fits a 4 GB GTX 1650; the mode measured in the 3-image test run).
  CPU = bf16 (~4.4 GB RAM).  fp32 is never used (~9 GB).  With fp16/8-bit the authors' published numbers may not
  hold any more; every answer is checked for NaN/inf.
"""
from __future__ import annotations

import argparse
import gc
import html
import io
import json
import math
import os
import sys
import threading
import time
import traceback
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PROMPT = "<image>Is this image real or fake? Answer only Real or Fake"     # README, section 3
MIN_PIXELS, MAX_PIXELS = "262144", "2408448"                              # README, section 4
SWIFT_COMMON = ["--max_new_tokens", "16", "--enable_thinking", "False", "--max_batch_size", "1",
                "--logprobs", "True", "--top_logprobs", "20", "--stream", "False"]
MODES = {
    "gpu": ("GPU fp16 + nén 8-bit (bitsandbytes)", ["--torch_dtype", "float16", "--quant_method", "bnb", "--quant_bits", "8"]),
    "cpu": ("CPU bf16 (không nén)", ["--torch_dtype", "bfloat16", "--device_map", "cpu"]),
}
JPEG_MAGIC, PNG_MAGIC = b"\xff\xd8\xff", b"\x89PNG\r\n\x1a\n"
ALLOWED_CTYPES = {"image/png", "image/jpeg", "application/octet-stream"}   # not CORS-"simple": browsers must preflight
APP_COMMAND = r"RUN_APP.bat --config research\configs\config_demo_defaker2.json"
INFO_PAGE = """<!doctype html><html lang="vi"><head><meta charset="utf-8"><title>Server DefakerOne2</title>
<style>body{{font-family:Segoe UI,Arial,sans-serif;max-width:760px;margin:32px auto;padding:0 16px;line-height:1.5}}
code{{background:#f1f3f5;padding:2px 6px;border-radius:4px}}</style></head><body>
<h1>Server DefakerOne2 (thử nghiệm) đang chạy</h1>
<p>Trạng thái mô hình: <b>{state}</b>.</p>
<p>Đây là địa chỉ để <b>ứng dụng</b> gửi ảnh tới (<code>POST /score</code>), <b>không phải giao diện</b> để mở bằng trình duyệt.</p>
<p>Mở giao diện: giữ cửa sổ server này, mở một cửa sổ dòng lệnh khác trong thư mục ứng dụng và chạy<br>
<code>{command}</code><br>Trình duyệt sẽ tự mở trang của ứng dụng; thẻ “DefakerOne2 (thử nghiệm)” hiện sau khi phân tích ảnh.</p>
<p>Trạng thái chi tiết (JSON): <a href="/health">/health</a></p></body></html>"""


def log(msg: str) -> None:
    print(time.strftime("%H:%M:%S ") + msg, flush=True)


class Engine:
    """ms-swift SwiftInfer built from the same arguments as the README command; one request at a time."""

    def __init__(self, repo: Path, model: Path, device: str):
        self.repo, self.model, self.device_choice = repo, model, device
        self.ready = self.loading = False
        self.error = None
        self.device = self.mode = None
        self.load_seconds = None
        self.requests = 0
        self.lock = threading.Lock()

    def status(self) -> dict:
        return {"ready": self.ready, "loading": self.loading, "error": self.error, "device": self.device,
                "mode": self.mode, "model": str(self.model), "load_seconds": self.load_seconds,
                "requests": self.requests, "prompt": PROMPT,
                "pixels_policy": {"MIN_PIXELS": int(os.environ["MIN_PIXELS"]), "MAX_PIXELS": int(os.environ["MAX_PIXELS"])}}

    def load(self, warmup: bool) -> None:
        self.loading = True
        t0 = time.perf_counter()
        try:
            import torch
            use_gpu = self.device_choice == "gpu" or (self.device_choice == "auto" and torch.cuda.is_available())
            if use_gpu and not torch.cuda.is_available():
                raise RuntimeError("Không thấy GPU CUDA (torch.cuda.is_available() = False). Chạy với --device cpu.")
            key = "gpu" if use_gpu else "cpu"
            self.mode, extra = MODES[key]
            self.device = (f"cuda:0 ({torch.cuda.get_device_name(0)})" if use_gpu else "cpu")
            sys.path.insert(0, str(self.repo))
            from logprob2score import logprobs_to_probs           # the authors' scoring code, unchanged
            self.logprobs_to_probs = logprobs_to_probs
            from swift.pipelines.infer.infer import SwiftInfer
            argv = ["--model", str(self.model), *extra, *SWIFT_COMMON]
            log(f"Nạp mô hình: {self.mode} trên {self.device} ...")
            self.infer = SwiftInfer(argv)
            self.request_config = self.infer.args.get_request_config()
            if self.request_config.stream or not self.request_config.logprobs or self.request_config.top_logprobs != 20:
                raise RuntimeError(f"Cấu hình suy luận khác README: {self.request_config}")
            gc.collect()
            if use_gpu:
                torch.cuda.empty_cache()
            self.load_seconds = round(time.perf_counter() - t0, 1)
            log(f"Đã nạp trọng số sau {self.load_seconds} s. {self._memory()}")
            if warmup:
                t1 = time.perf_counter()
                self.score(_warmup_png())
                self.requests = 0
                log(f"Chạy khởi động (warm-up) xong sau {time.perf_counter() - t1:.1f} s. {self._memory()}")
            self.ready = True
        except Exception as exc:  # keep the server up so /health and /score can report the problem
            self.error = f"{type(exc).__name__}: {exc}"
            log("LỖI khi nạp mô hình:\n" + traceback.format_exc())
        finally:
            self.loading = False

    def _memory(self) -> str:
        parts = []
        try:
            import psutil
            mi = psutil.Process().memory_info()
            text = f"RAM tiến trình {mi.rss / 2**30:.2f} GiB"
            if hasattr(mi, "private"):          # Windows: working set can be trimmed; private bytes cannot
                text += f" (đỉnh {mi.peak_wset / 2**30:.2f} GiB, private {mi.private / 2**30:.2f} GiB)"
            parts.append(text)
        except Exception:
            pass
        try:
            import torch
            if torch.cuda.is_available():
                parts.append(f"VRAM đang dùng {torch.cuda.memory_allocated() / 2**20:.0f} MiB, "
                             f"đỉnh {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
        except Exception:
            pass
        return "; ".join(parts)

    def score(self, image_bytes: bytes) -> dict:
        """One image -> README scores.  Raises ValueError for unreadable images, RuntimeError for bad outputs."""
        from PIL import Image, UnidentifiedImageError
        try:
            with Image.open(io.BytesIO(image_bytes)) as im:
                size = im.size
                im.verify()
        except (UnidentifiedImageError, OSError, SyntaxError) as exc:
            raise ValueError(f"Không đọc được ảnh: {exc}") from exc
        request = {"messages": [{"role": "user", "content": PROMPT}], "images": [{"bytes": image_bytes, "path": None}]}
        with self.lock:
            t0 = time.perf_counter()
            try:
                resp = self.infer.infer([request], self.request_config, use_tqdm=False)[0]
            except Exception as exc:
                if type(exc).__name__ == "OutOfMemoryError":
                    import torch
                    torch.cuda.empty_cache()
                    raise RuntimeError("GPU hết bộ nhớ khi chấm ảnh này.") from exc
                raise
            seconds = time.perf_counter() - t0
            self.requests += 1
        choice = resp.choices[0]
        lp = choice.logprobs if isinstance(choice.logprobs, dict) else {"content": choice.logprobs or []}
        content = lp.get("content") or []
        if not content:
            raise RuntimeError("Mô hình không trả về logprobs.")
        first = content[0]
        values = [first.get("logprob")] + [c.get("logprob") for c in first.get("top_logprobs", [])]
        if any(v is None or not math.isfinite(float(v)) for v in values):
            raise RuntimeError("Đầu ra của mô hình có NaN/inf (không dùng được điểm này).")
        p_fake, p_real = self.logprobs_to_probs({"logprobs": lp})
        if not (math.isfinite(p_fake) and math.isfinite(p_real)):
            raise RuntimeError("Điểm p_fake/p_real không hữu hạn.")
        return {"p_fake": p_fake, "p_real": p_real, "device": self.device, "mode": self.mode,
                "seconds": round(seconds, 3), "answer": first.get("token"),
                "response": choice.message.content,
                "top_logprobs": [[c.get("token"), c.get("logprob")] for c in first.get("top_logprobs", [])[:5]],
                "image_size_wh": list(size), "model_input_size_wh": _model_input_size(size),
                "pixels_policy": {"MIN_PIXELS": int(os.environ["MIN_PIXELS"]), "MAX_PIXELS": int(os.environ["MAX_PIXELS"])},
                "calibrated": False}


def _model_input_size(size_wh) -> list | None:
    """Size after the Qwen smart_resize (factor 32) applied by ms-swift: informational only."""
    try:
        from qwen_vl_utils.vision_process import smart_resize
        h, w = smart_resize(size_wh[1], size_wh[0], factor=32, min_pixels=int(os.environ["MIN_PIXELS"]),
                            max_pixels=int(os.environ["MAX_PIXELS"]))
        return [w, h]
    except Exception:
        return None


def _warmup_png() -> bytes:
    import numpy as np
    from PIL import Image
    yy, xx = np.mgrid[0:768, 0:1024]
    rgb = np.stack([xx % 256, yy % 256, (xx + yy) % 256], -1).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG")
    return buf.getvalue()


def make_handler(engine: Engine, allowed_hosts: set[str], max_bytes: int):
    class Handler(BaseHTTPRequestHandler):
        server_version = "DefakerOne2-addon/1"
        sys_version = ""

        def log_message(self, fmt, *args):  # requests are logged by do_POST in a shorter form
            pass

        def _json(self, code: int, data: dict) -> None:
            body = json.dumps(data, ensure_ascii=False).encode("utf-8")
            self.send_response(code)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.end_headers()
            self.wfile.write(body)

        def _host_ok(self) -> bool:     # DNS-rebinding protection
            return self.headers.get("Host", "") in allowed_hosts

        def do_GET(self):
            if not self._host_ok():
                return self._json(HTTPStatus.FORBIDDEN, {"error": "Host không hợp lệ."})
            path = self.path.split("?")[0]
            if path == "/health":
                return self._json(200, engine.status())
            if path in ("/", "/score"):     # someone opened the server address in a browser: explain, do not fail
                state = "sẵn sàng" if engine.ready else ("lỗi: " + engine.error if engine.error else "đang nạp")
                body = INFO_PAGE.format(state=html.escape(state), command=html.escape(APP_COMMAND)).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Cache-Control", "no-store")
                self.send_header("X-Content-Type-Options", "nosniff")
                self.send_header("Content-Security-Policy", "default-src 'none'; style-src 'unsafe-inline'")
                self.end_headers()
                self.wfile.write(body)
                return
            return self._json(404, {"error": "Không có trang này. Dùng POST /score hoặc GET /health."})

        def do_POST(self):
            if not self._host_ok():
                return self._json(HTTPStatus.FORBIDDEN, {"error": "Host không hợp lệ."})
            if self.path.split("?")[0] != "/score":
                return self._json(404, {"error": "Không có địa chỉ này. Dùng POST /score."})
            ctype = self.headers.get("Content-Type", "").split(";")[0].strip().lower()
            if ctype not in ALLOWED_CTYPES:
                return self._json(415, {"error": "Content-Type phải là image/png, image/jpeg hoặc application/octet-stream."})
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                length = -1
            if length <= 0:
                return self._json(400, {"error": "Chưa nhận được dữ liệu ảnh."})
            if length > max_bytes:
                return self._json(413, {"error": f"Ảnh vượt giới hạn {max_bytes // 2**20} MiB."})
            body = self.rfile.read(length)
            if not (body.startswith(JPEG_MAGIC) or body.startswith(PNG_MAGIC)):
                return self._json(415, {"error": "Chỉ nhận ảnh JPEG hoặc PNG."})
            if not engine.ready:
                msg = engine.error or ("Đang nạp mô hình, thử lại sau ít phút." if engine.loading else "Chưa nạp mô hình.")
                return self._json(503, {"error": msg})
            try:
                out = engine.score(body)
            except ValueError as exc:
                return self._json(422, {"error": str(exc)})
            except Exception as exc:
                log("LỖI khi chấm ảnh:\n" + traceback.format_exc())
                return self._json(500, {"error": f"{type(exc).__name__}: {exc}"})
            log(f"#{engine.requests} ảnh {out['image_size_wh'][0]}x{out['image_size_wh'][1]} -> "
                f"p_fake={out['p_fake']:.4f} ({out['answer']}) trong {out['seconds']:.1f} s")
            return self._json(200, out)

    return Handler


class LocalServer(ThreadingHTTPServer):
    """127.0.0.1 only.  On Windows SO_EXCLUSIVEADDRUSE (not SO_REUSEADDR) so a busy port fails clearly."""
    daemon_threads = True
    allow_reuse_address = os.name != "nt"

    def server_bind(self):
        import socket
        if os.name == "nt" and hasattr(socket, "SO_EXCLUSIVEADDRUSE"):
            self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_EXCLUSIVEADDRUSE, 1)
        super().server_bind()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Server DefakerOne2 cục bộ (chỉ 127.0.0.1) cho ImageForgeryDetection_v2.")
    ap.add_argument("--repo", type=Path, default=os.environ.get("DEFAKER2_REPO"),
                    help="thư mục mã nguồn DefakerOne2 (có logprob2score.py)")
    ap.add_argument("--model", type=Path, default=os.environ.get("DEFAKER2_MODEL"),
                    help="thư mục trọng số release/ (có config.json, model.safetensors)")
    ap.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    ap.add_argument("--port", type=int, default=int(os.environ.get("DEFAKER2_PORT", "8765")))
    ap.add_argument("--max-mb", type=int, default=256, help="giới hạn dung lượng một ảnh gửi tới (MiB)")
    ap.add_argument("--no-warmup", action="store_true", help="bỏ lượt chạy khởi động sau khi nạp")
    ap.add_argument("--verbose", action="store_true", help="in log chi tiết của ms-swift")
    args = ap.parse_args(argv)
    host = "127.0.0.1"
    if args.repo is None or not (Path(args.repo) / "logprob2score.py").is_file():
        print(f"[LỖI] Không thấy logprob2score.py trong --repo {args.repo}", file=sys.stderr)
        return 2
    if args.model is None or not (Path(args.model) / "config.json").is_file() \
            or not (Path(args.model) / "model.safetensors").is_file():
        print(f"[LỖI] Thư mục trọng số không hợp lệ (cần config.json và model.safetensors): {args.model}", file=sys.stderr)
        return 2
    # environment of the README command; must be set before torch / ms-swift are imported
    os.environ.setdefault("MIN_PIXELS", MIN_PIXELS)
    os.environ.setdefault("MAX_PIXELS", MAX_PIXELS)
    os.environ.setdefault("HF_HUB_OFFLINE", "1")          # local weights only, no download
    os.environ.setdefault("LOG_LEVEL", "INFO" if args.verbose else "WARNING")
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    else:
        os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    engine = Engine(Path(args.repo).resolve(), Path(args.model).resolve(), args.device)
    try:   # bind BEFORE loading: a busy port fails fast
        httpd = LocalServer((host, args.port), make_handler(engine, {f"{host}:{args.port}", f"localhost:{args.port}"},
                                                            args.max_mb * 2**20))
    except OSError as exc:
        print(f"[LỖI] Không mở được cổng {args.port}: {exc}. Có thể server DefakerOne2 khác đang chạy.", file=sys.stderr)
        return 2
    url = f"http://{host}:{args.port}"
    print("=" * 72)
    print(" DefakerOne2 (THỬ NGHIỆM): server cục bộ cho ImageForgeryDetection_v2")
    print(f" Địa chỉ để ỨNG DỤNG gửi ảnh: {url}/score  (không phải giao diện; trạng thái: {url}/health)")
    print(" Chỉ nghe ở 127.0.0.1: ảnh không rời khỏi máy. Đóng cửa sổ này để dừng.")
    print("=" * 72, flush=True)
    threading.Thread(target=_load_then_announce, args=(engine, not args.no_warmup, url), daemon=True).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
    return 0


def _load_then_announce(engine: Engine, warmup: bool, url: str) -> None:
    engine.load(warmup)
    if engine.ready:
        print("-" * 72)
        log(f"DA NAP XONG: server sẵn sàng tại {url}/score ({engine.mode}, {engine.device}).")
        print("Giữ cửa sổ này. Mở giao diện ở một cửa sổ dòng lệnh KHÁC, trong thư mục ứng dụng:", flush=True)
        print(f"    {APP_COMMAND}", flush=True)
        print("Trình duyệt sẽ tự mở trang của ứng dụng; thẻ 'DefakerOne2 (thử nghiệm)' hiện sau khi phân tích ảnh.", flush=True)
    else:
        log(f"KHÔNG NẠP ĐƯỢC MÔ HÌNH: {engine.error}")


if __name__ == "__main__":
    raise SystemExit(main())
