"""Research face pipeline: YuNet (OpenCV Zoo, MIT) detection + DeepfakeBench-style square crop (scale 1.3)."""
from pathlib import Path
import cv2
import numpy as np

YUNET = str(Path(__file__).resolve().parents[2] / "models" / "face_detection_yunet_2023mar.onnx")


class FaceFinder:
    def __init__(self, model=YUNET, score_threshold=0.7, max_side=1024):
        self.det = cv2.FaceDetectorYN.create(model, "", (320, 320), score_threshold, 0.3, 5000)
        self.max_side = max_side

    def find(self, rgb):
        h, w = rgb.shape[:2]
        s = min(1.0, self.max_side / max(h, w))
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if s < 1:
            img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
        self.det.setInputSize((img.shape[1], img.shape[0]))
        _, faces = self.det.detect(img)
        out = []
        if faces is not None:
            for f in faces:
                x, y, fw, fh = (f[:4] / s).tolist()
                out.append({"box": [x, y, fw, fh], "score": float(f[14]), "landmarks": (f[4:14] / s).reshape(5, 2).tolist()})
        out.sort(key=lambda d: -d["box"][2] * d["box"][3])
        return out


def square_crop(rgb, box, scale=1.3):
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
