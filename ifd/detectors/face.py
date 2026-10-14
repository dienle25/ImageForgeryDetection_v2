"""Face deepfake / face-swap detector: YuNet face detection (OpenCV Zoo, MIT) + LN-tuned CLIP ViT-L/14 classifier
("Unlocking the Hidden Potential of CLIP in Generalizable Deepfake Detection", Yermakov et al.; MIT;
HF yermandy/deepfake-detection; trained on FaceForensics++).

Pipeline (identical to the research evaluation that produced the thresholds):
  1. YuNet on the image (downscaled with INTER_AREA so the long side <= 1024), score >= 0.7, NMS 0.3.
  2. Keep the `max_faces` largest faces; square crop around each box, side = 1.3 x max(w, h), zero padding.
  3. CLIP preprocessing: PIL bicubic resize shorter side -> 224, centre crop 224, CLIP mean/std.
  4. p_fake = softmax(logits)[fake]; image score = max over checked faces.
No face found -> task 'not applicable' (the detector says nothing about the image).
"""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np
from PIL import Image

from ..errors import ForgeryError, model_load_error
from .base import DetectorResult, decide

MODEL_ID = "yermandy/deepfake-detection (CLIP ViT-L/14, LN-tuning, FF++) — ONNX; faces: YuNet 2023mar"
_MEAN = np.array([0.48145466, 0.4578275, 0.40821073], np.float32)
_STD = np.array([0.26862954, 0.26130258, 0.27577711], np.float32)


def clip_preprocess(rgb: np.ndarray) -> np.ndarray:
    im = Image.fromarray(rgb)
    w, h = im.size
    s = 224 / min(w, h)
    im = im.resize((max(224, round(w * s)), max(224, round(h * s))), Image.Resampling.BICUBIC)
    w, h = im.size
    left, top = (w - 224) // 2, (h - 224) // 2
    a = (np.asarray(im.crop((left, top, left + 224, top + 224)), np.float32) / 255 - _MEAN) / _STD
    return a.transpose(2, 0, 1)


def square_crop(rgb: np.ndarray, box, scale: float = 1.3):
    h, w = rgb.shape[:2]
    x, y, bw, bh = box
    cx, cy = x + bw / 2, y + bh / 2
    side = max(bw, bh) * scale
    x0, y0 = int(round(cx - side / 2)), int(round(cy - side / 2))
    x1, y1 = int(round(cx + side / 2)), int(round(cy + side / 2))
    pad = [max(0, -y0), max(0, y1 - h), max(0, -x0), max(0, x1 - w)]
    crop = rgb[max(0, y0):min(h, y1), max(0, x0):min(w, x1)]
    if any(pad):
        crop = cv2.copyMakeBorder(crop, pad[0], pad[1], pad[2], pad[3], cv2.BORDER_CONSTANT, value=0)
    return crop, [x0, y0, x1, y1]


class FaceFinder:
    def __init__(self, model_path, score_threshold: float = 0.7, max_side: int = 1024):
        try:
            self.det = cv2.FaceDetectorYN.create(str(model_path), "", (320, 320), score_threshold, 0.3, 5000)
        except Exception as exc:
            raise model_load_error("phát hiện khuôn mặt", model_path, exc) from exc
        self.max_side = max_side
        self._lock = threading.Lock()

    def find(self, rgb: np.ndarray) -> list[dict]:
        h, w = rgb.shape[:2]
        s = min(1.0, self.max_side / max(h, w))
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if s < 1:
            img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        with self._lock:
            self.det.setInputSize((img.shape[1], img.shape[0]))
            _, faces = self.det.detect(img)
        out = []
        if faces is not None:
            for f in faces:
                x, y, fw, fh = (f[:4] / s).tolist()
                out.append({"box": [x, y, fw, fh], "score": float(f[14])})
        out.sort(key=lambda d: -d["box"][2] * d["box"][3])
        return out


class FaceDeepfakeDetector:
    key = "face_clip"
    task = "face_deepfake"

    def __init__(self, onnx_path, face_model_path, settings: dict, threads: int = 0):
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
            raise model_load_error("deepfake khuôn mặt", onnx_path, exc) from exc
        if [i.name for i in self.session.get_inputs()] != ["face"] or \
                [o.name for o in self.session.get_outputs()] != ["logits"]:
            raise ForgeryError("File ONNX deepfake khuôn mặt không đúng giao diện (cần input 'face', output 'logits').")
        self.finder = FaceFinder(face_model_path, float(settings.get("face_score_threshold", 0.7)))
        self._lock = threading.Lock()

    def p_fake(self, crops: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
        x = np.stack([clip_preprocess(c) for c in crops]).astype(np.float32)
        with self._lock:
            lg = self.session.run(None, {"face": x})[0].astype(np.float64)
        e = np.exp(lg - lg.max(axis=1, keepdims=True))
        p = e / e.sum(axis=1, keepdims=True)
        return p[:, 1], lg

    def predict(self, rgb: np.ndarray, settings: dict | None = None) -> DetectorResult:
        settings = settings or self.settings
        th = settings["thresholds"]
        max_faces = int(settings.get("max_faces", 3))
        t0 = time.perf_counter()
        found = self.finder.find(rgb)
        faces = found[:max_faces]
        warnings = []
        if not faces:
            return DetectorResult(
                key=self.key, task=self.task, score=None, score_type="không có khuôn mặt được phát hiện",
                calibrated=False, decision="not_applicable", thresholds=th,
                details={"model": MODEL_ID, "faces_found": 0,
                         "note": "Không phát hiện khuôn mặt (YuNet, ngưỡng 0,7) → không kiểm tra deepfake khuôn mặt."},
                timing={"face_detection_and_inference": time.perf_counter() - t0})
        if len(found) > max_faces:
            warnings.append(f"Ảnh có {len(found)} khuôn mặt; chỉ {max_faces} khuôn mặt lớn nhất được kiểm tra.")
        crops, rects = [], []
        for f in faces:
            c, r = square_crop(rgb, f["box"])
            crops.append(c)
            rects.append(r)
        p, lg = self.p_fake(crops)
        score = float(p.max())
        per_face = []
        for f, r, pf, l in zip(faces, rects, p, lg):
            per_face.append({"box_xywh": [round(v, 1) for v in f["box"]], "crop_xyxy": r,
                             "detection_score": round(f["score"], 4), "score": float(pf),
                             "logits_real_fake": [float(l[0]), float(l[1])],
                             "decision": decide(float(pf), th["low"], th["high"])})
        small = [pf for pf in per_face if min(pf["box_xywh"][2:]) < 64]
        if small:
            warnings.append("Có khuôn mặt nhỏ (< 64 px): độ tin cậy kiểm tra deepfake thấp hơn.")
        return DetectorResult(
            key=self.key, task=self.task, score=score,
            score_type="CLIP deepfake: max softmax(fake) trên các khuôn mặt — điểm thô, CHƯA hiệu chỉnh xác suất",
            calibrated=False, decision=decide(score, th["low"], th["high"]), thresholds=th,
            heatmap=None, heatmap_type=None,
            details={"model": MODEL_ID, "faces_found": len(found), "faces_checked": len(faces), "faces": per_face,
                     "transform": {"face_detector": "YuNet 2023mar, score ≥ 0.7, NMS 0.3, cạnh dài ≤ 1024 khi dò",
                                   "crop": "vuông, cạnh = 1.3 × max(w,h), đệm 0", "classifier_input": "224×224 CLIP"}},
            warnings=warnings,
            timing={"face_detection_and_inference": time.perf_counter() - t0})
