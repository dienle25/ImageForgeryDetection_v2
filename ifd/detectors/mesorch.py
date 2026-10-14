"""Mesorch (Zhu et al., AAAI 2025; code MIT, official checkpoint) on ONNX Runtime — local manipulation map.

Preprocessing = official test pipeline: Resize(512x512, cv2.INTER_LINEAR on uint8 RGB) -> /255 -> ImageNet mean/std.
The network outputs a sigmoid map at 512x512 with a FIXED [0,1] scale.  Mesorch has no image-level head, so the
image score is a fixed statistic of that native 512x512 map (chosen on the development split, see config).
The display heatmap is the 512x512 map resized back to the displayed image size (cv2.INTER_LINEAR); no per-image
rescaling of the map is ever applied.
"""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np

from ..errors import ForgeryError, model_load_error
from .base import DetectorResult, decide

MODEL_ID = "Mesorch (Zhu et al., AAAI 2025) official checkpoint mesorch-98.pth — ONNX"
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)
SIZE = 512


def preprocess(rgb: np.ndarray) -> np.ndarray:
    x = cv2.resize(np.ascontiguousarray(rgb), (SIZE, SIZE), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    x = (x - _MEAN) / _STD
    return np.ascontiguousarray(x.transpose(2, 0, 1)[None])


def map_statistic(m: np.ndarray, name: str) -> float:
    flat = np.sort(m.ravel())[::-1]
    n = flat.size
    if name == "max":
        return float(flat[0])
    if name == "p999":
        return float(flat[max(0, int(n * 0.001) - 1)])
    if name == "top1pct_mean":
        return float(flat[: max(1, n // 100)].mean())
    raise ForgeryError(f"Thống kê điểm ảnh không hỗ trợ: {name}")


class MesorchDetector:
    key = "mesorch"
    task = "local_manipulation"

    def __init__(self, onnx_path, settings: dict, threads: int = 0):
        import onnxruntime as ort
        self.settings = settings
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        if threads:
            opts.intra_op_num_threads = int(threads)
        opts.inter_op_num_threads = 1
        try:
            self.session = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
        except Exception as exc:
            raise model_load_error("Mesorch", onnx_path, exc) from exc
        if [i.name for i in self.session.get_inputs()] != ["image"] or \
                [o.name for o in self.session.get_outputs()] != ["mask_prob"]:
            raise ForgeryError("File ONNX Mesorch không đúng giao diện (cần input 'image', output 'mask_prob').")
        self._lock = threading.Lock()

    def native_map(self, rgb: np.ndarray) -> np.ndarray:
        x = preprocess(rgb)
        with self._lock:
            out = self.session.run(None, {"image": x})[0]
        return out[0, 0].astype(np.float32)

    def predict(self, rgb: np.ndarray, settings: dict | None = None) -> DetectorResult:
        settings = settings or self.settings
        h, w = rgb.shape[:2]
        t0 = time.perf_counter()
        m = self.native_map(rgb)
        if not np.isfinite(m).all():
            raise ForgeryError("Mô hình trả về số không hữu hạn; không xuất kết quả.")
        stat = settings.get("image_score", "max")
        score = map_statistic(m, stat)
        heat = np.clip(cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR), 0.0, 1.0).astype(np.float32)
        infer = time.perf_counter() - t0
        th = settings["thresholds"]
        return DetectorResult(
            key=self.key, task=self.task, score=score,
            score_type=f"Mesorch: '{stat}' của bản đồ sigmoid 512×512 — điểm thô, CHƯA hiệu chỉnh thành xác suất",
            calibrated=False, decision=decide(score, th["low"], th["high"]), thresholds=th,
            heatmap=heat, heatmap_type="Mesorch sigmoid map (điểm nghi vấn từng pixel, thang cố định [0,1], chưa hiệu chỉnh)",
            details={"model": MODEL_ID, "image_score_statistic": stat,
                     "transform": {"mode": "global_resize", "model_input_hw": [SIZE, SIZE], "resample": "cv2.INTER_LINEAR",
                                   "aspect_ratio_preserved": False, "normalization": "ImageNet mean/std",
                                   "rationale": "official Mesorch test pipeline (albumentations Resize 512 + Normalize)",
                                   "heatmap_restore": "cv2.INTER_LINEAR 512×512 → kích thước ảnh hiển thị"},
                     "native_map_stats": {"max": float(m.max()), "mean": float(m.mean()),
                                          "fraction_ge_pixel_threshold": float((m >= th["pixel"]).mean())},
                     "heatmap_stats": {"min": float(heat.min()), "max": float(heat.max()), "mean": float(heat.mean()),
                                       "fraction_ge_pixel_threshold": float((heat >= th["pixel"]).mean())}},
            timing={"inference_and_resize": infer})
