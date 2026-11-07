"""DefakerOne2 (inclusionAI/DefakerOne2, Qwen3.5-2B) — EXPERIMENTAL whole-image real/fake probability.

The model needs PyTorch and ~4 GB of GPU memory, so it is NOT part of the app runtime.  It runs in its own
process (addons/defaker2/defaker2_server.py, conda env `defaker2`) bound to 127.0.0.1; this detector only posts
the decoded image there with the Python standard library (no new dependency in the app runtime).

* Score = p_fake of the model's first answer token, computed by the server exactly like the README +
  logprob2score.py of the DefakerOne2 repository.  Raw, NOT calibrated, no validated thresholds.
* The result never decides anything: decision "experimental", in_summary=False (own card, not in the overall
  decision, not in the decision-config hash).
* Server not running / still loading / timeout / bad answer -> decision "unavailable" ("Chưa khả dụng"); the app
  keeps working with its evaluated detectors.
* Proxies are disabled for this connection: the image goes to 127.0.0.1 only, never through a system proxy.
"""
from __future__ import annotations

import io
import json
import math
import socket
import time
import urllib.error
import urllib.request

import numpy as np
from PIL import Image

from .base import DetectorResult, decide

MODEL_ID = ("DefakerOne2 release (inclusionAI/DefakerOne2; trọng số Hugging Face qcf568/DefakerOne2, thư mục "
            "release/; nền Qwen3.5-2B) — chạy ngoài ứng dụng qua server cục bộ")
SCORE_TYPE = ("DefakerOne2 p_fake — xác suất của token 'Fake' ở câu trả lời đầu tiên, chuẩn hóa Real/Fake theo "
              "logprob2score.py của tác giả. Điểm thô, CHƯA hiệu chỉnh, chưa có ngưỡng kiểm chứng. "
              "THỬ NGHIỆM: không tính vào kết luận chung.")
START_HINT = "Chạy addons\\defaker2\\CHAY_DEFAKER2.bat, chờ dòng 'DA NAP XONG' rồi phân tích lại."
VALIDATION = {
    "status": "experimental",
    "short_vi": "Thử nghiệm: chưa kiểm định trên dữ liệu của nhóm. Không tính vào kết luận chung.",
    "text_vi": ("DefakerOne2 là mô hình thử nghiệm chạy ngoài ứng dụng (server cục bộ 127.0.0.1). Điểm p_fake là "
                "điểm thô chưa hiệu chỉnh. Số liệu công bố của tác giả chưa được kiểm chứng độc lập. Khi chạy nén "
                "8-bit / fp16, số liệu đó có thể không còn đúng. Điểm này KHÔNG được tính vào kết luận chung."),
}
PRIMARY_SCORE_TYPE = ("DefakerOne2 p_fake — xác suất của token 'Fake' ở câu trả lời đầu tiên, chuẩn hóa Real/Fake theo "
                      "logprob2score.py của tác giả. MÔ HÌNH CHÍNH: kết luận theo cách của tác giả — mô hình trả lời "
                      "'Fake' (p_fake ≥ 0,5) là 'Nghi vấn'. Điểm thô, chưa hiệu chỉnh, chưa kiểm chứng trên dữ liệu của nhóm.")
PRIMARY_VALIDATION = {
    "status": "author_rule",
    "short_vi": "Mô hình chính. Ngưỡng 0,5 theo cách tác giả ra kết luận (token Real/Fake); chưa kiểm định trên dữ liệu của nhóm.",
    "text_vi": ("DefakerOne2 là mô hình chính: kết luận chung theo câu trả lời Real/Fake của mô hình (p_fake ≥ 0,5 ⇒ "
                "'Nghi vấn'), đúng cách repo tác giả tính accuracy. Repo tác giả không công bố ngưỡng nào khác. Số liệu "
                "công bố chưa kiểm chứng độc lập; chạy nén 8-bit/fp16 có thể làm số liệu đó không còn đúng."),
}
IMAGE_SENT_VI = ("Ảnh đã đọc của ứng dụng (đã xoay theo EXIF, nền trong suốt ghép lên nền trắng), gửi dạng PNG "
                 "không mất dữ liệu tới server cục bộ; server tự thu nhỏ theo MIN_PIXELS/MAX_PIXELS như README.")

# direct connection only (ignores HTTP_PROXY / Windows proxy settings)
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def model_info(settings: dict) -> dict:
    """Entry for the report's 'models' block (there is no model file inside the app to hash)."""
    return {"file": None, "sha256": None, "source": MODEL_ID, "endpoint": settings.get("url"),
            "license_code": "Apache-2.0 (mã nguồn repo inclusionAI/DefakerOne2)",
            "license_weights": "Chưa rõ: README chưa công bố giấy phép trọng số. Không phát tán kèm trọng số.",
            "note": "Mô hình thử nghiệm chạy ngoài ứng dụng; trọng số không nằm trong gói và không được kiểm SHA-256 tại đây."}


def encode_png(rgb: np.ndarray) -> bytes:
    """Lossless, fast PNG of the decoded image (compress_level=1: ~3× faster than the default, same pixels)."""
    buf = io.BytesIO()
    Image.fromarray(rgb).save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def _error_text(raw: bytes) -> str:
    try:
        data = json.loads(raw.decode("utf-8"))
        if isinstance(data, dict) and data.get("error"):
            return str(data["error"])[:300]
    except (ValueError, UnicodeDecodeError):
        pass
    return raw[:200].decode("utf-8", "replace")


class DefakerTwoHTTPDetector:
    key = "defaker2"
    task = "whole_image_authenticity"

    def __init__(self, settings: dict):
        self.settings = settings

    def _settings(self, settings: dict | None) -> tuple[str, float]:
        s = settings or self.settings
        return s["url"], float(s.get("timeout_seconds", 120))

    def health(self, timeout: float = 1.0) -> dict:
        """GET <server>/health: {"state": "ready" | "loading" | "error" | "down", ...} for the web page."""
        url, _ = self._settings(None)
        health_url = url.rsplit("/", 1)[0] + "/health"
        try:
            with _OPENER.open(health_url, timeout=timeout) as r:
                data = json.loads(r.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            return {"state": "error", "message": f"HTTP {exc.code}"}
        except (OSError, ValueError):
            return {"state": "down", "message": "Chưa chạy server DefakerOne2. " + START_HINT}
        if not isinstance(data, dict):
            return {"state": "error", "message": "Phản hồi /health không hợp lệ."}
        state = "ready" if data.get("ready") else ("error" if data.get("error") else "loading")
        return {"state": state, "device": data.get("device"), "mode": data.get("mode"),
                "message": data.get("error")}

    @property
    def primary(self) -> bool:
        return self.settings.get("role") == "primary"

    def _result(self, t0: float, score=None, decision="unavailable", warnings=None, details=None, timing=None):
        det = {"model": MODEL_ID, "server_url": self._settings(None)[0],
               "validation": dict(PRIMARY_VALIDATION if self.primary else VALIDATION),
               "image_sent": IMAGE_SENT_VI, "localization": "Không có bản đồ vùng: mô hình chỉ cho điểm toàn ảnh."}
        det.update(details or {})
        tim = {"total": time.perf_counter() - t0}
        tim.update(timing or {})
        if self.primary and score is not None:
            th = self.settings["thresholds"]
            decision = decide(score, th["low"], th["high"])
        return DetectorResult(key=self.key, task=self.task, score=score,
                              score_type=PRIMARY_SCORE_TYPE if self.primary else SCORE_TYPE, calibrated=False,
                              decision=decision, thresholds=dict(self.settings.get("thresholds", {})) if self.primary else {},
                              heatmap=None, heatmap_type=None, details=det, warnings=list(warnings or []), timing=tim,
                              in_summary=self.primary, primary=self.primary)

    def predict(self, rgb: np.ndarray, settings: dict | None = None) -> DetectorResult:
        url, timeout = self._settings(settings)
        t0 = time.perf_counter()
        body = encode_png(rgb)
        t_enc = time.perf_counter() - t0
        req = urllib.request.Request(url, data=body, method="POST",
                                     headers={"Content-Type": "image/png", "Accept": "application/json"})
        t1 = time.perf_counter()
        try:
            with _OPENER.open(req, timeout=timeout) as r:
                raw = r.read()
        except urllib.error.HTTPError as exc:
            msg = _error_text(exc.read() or b"")
            if exc.code == 503:
                text = f"Server DefakerOne2 chưa sẵn sàng: {msg}"
            else:
                text = f"Server DefakerOne2 báo lỗi (HTTP {exc.code}): {msg}"
            return self._result(t0, warnings=[text], timing={"encode_png": t_enc})
        except (TimeoutError, socket.timeout):
            return self._result(t0, warnings=[f"Server DefakerOne2 không trả lời trong {timeout:g} giây; bỏ qua "
                                              "bộ phát hiện thử nghiệm này cho ảnh này."], timing={"encode_png": t_enc})
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (TimeoutError, socket.timeout)):
                text = f"Server DefakerOne2 không trả lời trong {timeout:g} giây."
            else:
                text = f"Chưa kết nối được server DefakerOne2 tại {url}. {START_HINT}"
            return self._result(t0, warnings=[text], timing={"encode_png": t_enc})
        except OSError as exc:  # connection reset/aborted while reading
            return self._result(t0, warnings=[f"Mất kết nối với server DefakerOne2 ({type(exc).__name__})."],
                                timing={"encode_png": t_enc})
        rtt = time.perf_counter() - t1
        try:
            data = json.loads(raw.decode("utf-8"))
            p_fake, p_real = float(data["p_fake"]), float(data["p_real"])
        except (ValueError, KeyError, TypeError, UnicodeDecodeError):
            return self._result(t0, warnings=["Server DefakerOne2 trả về dữ liệu không hợp lệ."],
                                timing={"encode_png": t_enc, "http_round_trip": rtt})
        if not (math.isfinite(p_fake) and math.isfinite(p_real) and 0.0 <= p_fake <= 1.0):
            return self._result(t0, warnings=["Server DefakerOne2 trả về điểm không hợp lệ (NaN/inf hoặc ngoài [0, 1])."],
                                timing={"encode_png": t_enc, "http_round_trip": rtt})
        details = {"p_fake": p_fake, "p_real": p_real, "answer": data.get("answer"),
                   "device": data.get("device"), "mode": data.get("mode"),
                   "first_token_top_logprobs": data.get("top_logprobs"),
                   "model_input_size_wh": data.get("model_input_size_wh"),
                   "transform": {"sent": "PNG không mất dữ liệu của ảnh đã đọc",
                                 "server_resize": data.get("pixels_policy")}}
        timing = {"encode_png": t_enc, "http_round_trip": rtt, "server_inference": data.get("seconds")}
        return self._result(t0, score=p_fake, decision="experimental", details=details, timing=timing)
