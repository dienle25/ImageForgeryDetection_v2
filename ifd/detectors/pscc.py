"""PSCC-Net (Liu et al., TCSVT 2022; MIT) on ONNX Runtime with an explicit preprocessing policy."""
from __future__ import annotations

import threading
import time

import numpy as np

from ..errors import ForgeryError, model_load_error
from ..resample import interp_align_corners, resize_rgb
from .base import DetectorResult, decide

MODEL_ID = "PSCC-Net@53e5ff77d8dc5feddda060cd085f9b765761f816 (ONNX, BN folded)"


class PSCCDetector:
    key = "pscc_net"
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
        except Exception as exc:  # corrupt / incompatible model file
            raise model_load_error("PSCC-Net", onnx_path, exc) from exc
        names = [o.name for o in self.session.get_outputs()]
        if names != ["mask", "logits"] or [i.name for i in self.session.get_inputs()] != ["image"]:
            raise ForgeryError("File ONNX PSCC-Net không đúng giao diện (cần input 'image', output 'mask','logits').")
        self._lock = threading.Lock()

    def _run(self, rgb_u8: np.ndarray):
        x = (rgb_u8.astype(np.float32) / 255.0).transpose(2, 0, 1)[None]
        with self._lock:
            mask, logits = self.session.run(None, {"image": np.ascontiguousarray(x)})
        lg = logits[0].astype(np.float64)
        e = np.exp(lg - lg.max())
        return float(e[1] / e.sum()), lg.tolist(), mask[0, 0]

    def predict(self, rgb: np.ndarray, settings: dict | None = None) -> DetectorResult:
        settings = settings or self.settings
        p = settings["policy"]
        h, w = rgb.shape[:2]
        t0 = time.perf_counter()
        if p["mode"] == "global_resize":
            size = int(p.get("size", 256))
            th, tw = size, size
            work = resize_rgb(rgb, (th, tw), p.get("resample", "bicubic"))
            score, logits, m = self._run(work)
            heat = interp_align_corners(interp_align_corners(m[None], (th, tw)), (h, w))[0]
            transform = {"mode": "global_resize", "model_input_hw": [th, tw], "resample": p.get("resample", "bicubic"),
                         "aspect_ratio_preserved": False,
                         "rationale": "matches the official training loader (PIL bicubic resize of every image to 256x256)"}
        elif p["mode"] == "bounded_native":
            max_side = int(p.get("max_side", 512))
            s = min(1.0, max_side / max(h, w))
            rh, rw = max(1, round(h * s)), max(1, round(w * s))
            work = resize_rgb(rgb, (rh, rw), p.get("resample", "bilinear"))
            ph, pw = (-rh) % 8, (-rw) % 8
            work = np.pad(work, ((0, ph), (0, pw), (0, 0)), mode="edge")
            score, logits, m = self._run(work)
            full = interp_align_corners(m[None], (rh + ph, rw + pw))[:, :rh, :rw]
            heat = interp_align_corners(full, (h, w))[0]
            transform = {"mode": "bounded_native", "max_side": max_side, "model_input_hw": [rh + ph, rw + pw],
                         "resized_hw": [rh, rw], "padding_bottom_right": [ph, pw]}
        else:
            raise ForgeryError(f"Chính sách tiền xử lý PSCC không hỗ trợ: {p['mode']}")
        infer = time.perf_counter() - t0
        if not np.isfinite(score) or not np.isfinite(heat).all():
            raise ForgeryError("Mô hình trả về số không hữu hạn; không xuất kết quả.")
        heat = np.clip(heat, 0, 1).astype(np.float32)
        th = settings["thresholds"]
        return DetectorResult(
            key=self.key, task=self.task, score=score,
            score_type="PSCC-Net softmax(class 'forged') — điểm thô của mô hình, CHƯA hiệu chỉnh thành xác suất",
            calibrated=False, decision=decide(score, th["low"], th["high"]), thresholds=th,
            heatmap=heat, heatmap_type="PSCC-Net NLCDetection sigmoid map (điểm nghi vấn từng pixel, chưa hiệu chỉnh)",
            details={"model": MODEL_ID, "logits_authentic_forged": logits, "transform": transform,
                     "heatmap_stats": {"min": float(heat.min()), "max": float(heat.max()), "mean": float(heat.mean()),
                                       "fraction_ge_pixel_threshold": float((heat >= th["pixel"]).mean())}},
            timing={"inference_and_resize": infer})
