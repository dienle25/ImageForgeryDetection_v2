"""Configuration (config.json) with strict validation.  Paths are relative to the config file."""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from .errors import ForgeryError

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = PROJECT_ROOT / "config.json"


def _num(value, name, lo=None, hi=None, integer=False):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or (integer and not isinstance(value, int)):
        raise ForgeryError(f"Cấu hình '{name}' phải là số{' nguyên' if integer else ''}.")
    if not math.isfinite(value):
        raise ForgeryError(f"Cấu hình '{name}' phải là số hữu hạn.")
    if (lo is not None and value < lo) or (hi is not None and value > hi):
        raise ForgeryError(f"Cấu hình '{name}' phải trong khoảng [{lo}, {hi}].")
    return value


def validate_thresholds(th: dict, name: str) -> dict:
    low = _num(th.get("low"), f"{name}.low", 0, 1)
    high = _num(th.get("high"), f"{name}.high", 0, 1)
    if not low <= high:
        raise ForgeryError(f"Ngưỡng {name}: cần low ≤ high.")
    if "pixel" in th:
        _num(th["pixel"], f"{name}.pixel", 0, 1)
    return th


class Config:
    def __init__(self, data: dict, base_dir: Path):
        self.data = data
        self.base_dir = base_dir
        self.validate()

    def validate(self) -> "Config":
        d = self.data
        lim = d.get("limits", {})
        _num(lim.get("max_file_mb"), "limits.max_file_mb", 0.1, 200)
        _num(lim.get("max_pixels"), "limits.max_pixels", 64, 200_000_000, integer=True)
        _num(lim.get("min_side"), "limits.min_side", 8, 4096, integer=True)
        _num(d.get("runtime", {}).get("threads", 0), "runtime.threads", 0, 256, integer=True)
        _num(d.get("display", {}).get("overlay_alpha", 0.55), "display.overlay_alpha", 0, 1)
        dets = d.get("detectors")
        if not isinstance(dets, dict) or not dets:
            raise ForgeryError("Cấu hình thiếu mục 'detectors'.")
        known = {"pscc_onnx", "mesorch_onnx", "commfor_onnx", "face_clip_onnx"}
        for key, det in dets.items():
            if not isinstance(det, dict):
                raise ForgeryError(f"Cấu hình detector '{key}' không hợp lệ.")
            if det.get("type") not in known:
                raise ForgeryError(f"Detector '{key}': loại '{det.get('type')}' không hỗ trợ.")
            if not isinstance(det.get("model"), str) or not det["model"]:
                raise ForgeryError(f"Detector '{key}': thiếu đường dẫn 'model'.")
            if det["type"] == "face_clip_onnx":
                if not isinstance(det.get("face_model"), str) or not det["face_model"]:
                    raise ForgeryError(f"Detector '{key}': thiếu 'face_model' (mô hình phát hiện khuôn mặt).")
                _num(det.get("max_faces", 3), f"{key}.max_faces", 1, 20, integer=True)
                _num(det.get("face_score_threshold", 0.7), f"{key}.face_score_threshold", 0.05, 0.99)
            if det["type"] == "mesorch_onnx" and det.get("image_score", "max") not in ("max", "p999", "top1pct_mean"):
                raise ForgeryError(f"Detector '{key}': image_score phải là max, p999 hoặc top1pct_mean.")
            if det.get("enabled"):
                validate_thresholds(det.get("thresholds", {}), key)
        return self

    @property
    def limits(self) -> dict:
        return self.data["limits"]

    @property
    def threads(self) -> int:
        return int(self.data.get("runtime", {}).get("threads", 0))

    @property
    def overlay_alpha(self) -> float:
        return float(self.data.get("display", {}).get("overlay_alpha", 0.55))

    def detector(self, key: str) -> dict:
        return self.data["detectors"][key]

    def path(self, relative: str) -> Path:
        p = Path(relative)
        return p if p.is_absolute() else (self.base_dir / p)

    def with_threshold_overrides(self, key: str, **values) -> "Config":
        data = copy.deepcopy(self.data)
        th = data["detectors"][key]["thresholds"]
        for k, v in values.items():
            if v is not None:
                th[k] = v
                th["status"] = "user override (không phải ngưỡng đã kiểm chứng)"
        return Config(data, self.base_dir)

    def to_json(self) -> dict:
        return copy.deepcopy(self.data)


def load_config(path: str | Path = DEFAULT_CONFIG) -> Config:
    path = Path(path).resolve()
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ForgeryError(f"Không đọc được cấu hình {path.name}: {exc}") from exc
    if not isinstance(data, dict):
        raise ForgeryError("Cấu hình phải là JSON object.")
    return Config(data, path.parent)
