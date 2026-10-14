"""Community Forensics ViT-S/16@384 (Park & Owens, CVPR 2025; code+weights MIT) — whole-image AI-generation score.

Preprocessing reproduces the official eval transform: Resize(shorter side=440, bilinear) -> CenterCrop(384)
-> [0,1] -> ImageNet mean/std.  Output: sigmoid(logit) = raw "AI-generated" score (uncalibrated).
This detector does NOT localize; it never produces a heatmap.
"""
from __future__ import annotations

import threading
import time

import numpy as np
from PIL import Image

from ..errors import model_load_error
from .base import DetectorResult, decide

MODEL_ID = "Community Forensics commfor-model-384 (OwensLab, HF sha 6076002b; ViT-S/16@384) — ONNX"
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def preprocess(rgb: np.ndarray, resize: int = 440, crop: int = 384) -> np.ndarray:
    im = Image.fromarray(rgb)
    w, h = im.size
    if w <= h:
        nw, nh = resize, int(resize * h / w)
    else:
        nw, nh = int(resize * w / h), resize
    im = im.resize((nw, nh), Image.Resampling.BILINEAR)
    left = int(round((nw - crop) / 2.0))
    top = int(round((nh - crop) / 2.0))
    a = np.asarray(im.crop((left, top, left + crop, top + crop)), dtype=np.float32) / 255.0
    return ((a - _MEAN) / _STD).transpose(2, 0, 1)[None].astype(np.float32)


class CommForDetector:
    key = "commfor_aigen"
    task = "ai_generated"

    def __init__(self, onnx_path, settings: dict, threads: int = 0):
        import onnxruntime as ort
        self.settings = settings
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        if threads:
            opts.intra_op_num_threads = int(threads)
        try:
            self.session = ort.InferenceSession(str(onnx_path), sess_options=opts, providers=["CPUExecutionProvider"])
        except Exception as exc:
            raise model_load_error("ảnh sinh AI", onnx_path, exc) from exc
        self._lock = threading.Lock()

    def raw_logit(self, rgb: np.ndarray) -> float:
        x = preprocess(rgb)
        with self._lock:
            out = self.session.run(None, {"image": x})[0]
        return float(out.reshape(-1)[0])

    def predict(self, rgb: np.ndarray, settings: dict | None = None) -> DetectorResult:
        settings = settings or self.settings
        t0 = time.perf_counter()
        logit = self.raw_logit(rgb)
        score = float(1.0 / (1.0 + np.exp(-logit)))
        th = settings["thresholds"]
        return DetectorResult(
            key=self.key, task=self.task, score=score,
            score_type="Community Forensics sigmoid(logit) — điểm thô 'ảnh sinh bằng AI', CHƯA hiệu chỉnh xác suất",
            calibrated=False, decision=decide(score, th["low"], th["high"]), thresholds=th,
            heatmap=None, heatmap_type=None,
            details={"model": MODEL_ID, "logit": logit,
                     "transform": {"resize_shorter_side": 440, "center_crop": 384, "normalization": "ImageNet mean/std",
                                   "note": "chỉ nhìn vùng trung tâm 384×384 sau khi thu nhỏ; không định vị vùng"}},
            timing={"inference_and_resize": time.perf_counter() - t0})
