"""Research adapter: CAT-Net v2 (Kwon et al., IJCV 2022; code Apache-2.0, weights CC-BY-4.0 since 2026-08-05).

Reproduces Splicing/data/AbstractDataset._create_tensor + dataset_arbitrary.get_tamp for inference:
* non-JPEG input -> re-encode JPEG quality 100, 4:4:4 (as the official 'arbitrary' dataset does); RGB AND DCT read from that file
* RGB: (x-127.5)/127.5; padded to multiple of 8 with 127.5
* DCT: Y-channel quantized coefficients (jpeglib instead of jpegio; same block layout), zero-padded;
  DCTvol: 21 one-hot channels (|c|==0..19, |c|>=20); qtable = Y quantization table (natural order)
* output: softmax(logits)[1] at 1/4 resolution, upsampled bilinearly to the image size.
"""
import io, os, sys, tempfile
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

UP = Path(os.environ.get("IFD_UPSTREAM", Path(__file__).resolve().parents[3] / "upstream")) / "catnet"


def dct_volume(coef: np.ndarray, T: int = 20) -> np.ndarray:
    a = np.abs(coef)
    vol = np.zeros((T + 1,) + coef.shape, np.float32)
    vol[0] = coef == 0
    for i in range(1, T):
        vol[i] = a == i
    vol[T] = a >= T
    return vol


def jpeg_inputs(jpeg_bytes: bytes):
    import jpeglib
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tf:
        tf.write(jpeg_bytes)
        tmp = tf.name
    try:
        d = jpeglib.read_dct(tmp)
        y = np.asarray(d.Y)                       # (hb, wb, 8, 8)
        qt = np.asarray(d.qt)[int(np.asarray(d.quant_tbl_no)[0])].astype(np.float32)
        rgb = np.asarray(Image.open(tmp).convert("RGB"))
    finally:
        os.unlink(tmp)
    hb, wb = y.shape[:2]
    coef = y.transpose(0, 2, 1, 3).reshape(hb * 8, wb * 8).astype(np.float32)
    return rgb, coef, qt


class CatNetRunner:
    def __init__(self, ckpt: str, threads: int = 2):
        sys.path.insert(0, str(UP))
        torch.set_num_threads(threads)
        from lib.config import config, update_config  # noqa
        import argparse
        cwd = os.getcwd()
        os.chdir(UP)
        try:
            update_config(config, argparse.Namespace(cfg="experiments/CAT_full.yaml", opts=[]))
            config.defrost(); config.MODEL.PRETRAINED_RGB = ""; config.MODEL.PRETRAINED_DCT = ""; config.freeze()
            from lib.models import network_CAT
            self.model = network_CAT.get_seg_model(config)
        finally:
            os.chdir(cwd)
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        sd = ck["state_dict"] if "state_dict" in ck else ck
        sd = {(k[6:] if k.startswith("model.") else k): v for k, v in sd.items()}
        res = self.model.load_state_dict(sd, strict=True)
        self.load_report = {"missing": list(res.missing_keys), "unexpected": list(res.unexpected_keys), "epoch": ck.get("epoch")}
        self.model.eval()

    @torch.inference_mode()
    def predict(self, rgb_display: np.ndarray, path=None):
        """rgb_display is used only for output size; the network reads the JPEG file (or a Q100 re-encode)."""
        h, w = rgb_display.shape[:2]
        raw = Path(path).read_bytes() if path else None
        is_jpeg = raw is not None and raw[:2] == b"\xff\xd8"
        if is_jpeg:
            # EXIF-rotated JPEGs: model works in stored orientation; we rotate the map afterwards if needed
            pass
        else:
            buf = io.BytesIO(); Image.fromarray(rgb_display).save(buf, format="JPEG", quality=100, subsampling=0); raw = buf.getvalue()
        rgb, coef, qt = jpeg_inputs(raw)
        ih, iw = rgb.shape[:2]
        ch, cw = -(-ih // 8) * 8, -(-iw // 8) * 8
        rgbp = np.full((ch, cw, 3), 127.5, np.float32); rgbp[:ih, :iw] = rgb
        coefp = np.zeros((max(ch, coef.shape[0]), max(cw, coef.shape[1])), np.float32)
        coefp[:coef.shape[0], :coef.shape[1]] = coef
        coefp = coefp[:ch, :cw]
        x = np.concatenate([((rgbp - 127.5) / 127.5).transpose(2, 0, 1), dct_volume(coefp)], 0)[None]
        logits = self.model(torch.from_numpy(x), torch.from_numpy(qt[None, None]))
        prob = F.softmax(logits, dim=1)[:, 1:2]
        full = F.interpolate(prob, size=(ch, cw), mode="bilinear", align_corners=False)[0, 0, :ih, :iw].numpy()
        if (ih, iw) != (h, w):
            if (ih, iw) == (w, h):   # EXIF 90/270 rotation: approximate by matching display orientation
                full = np.ascontiguousarray(np.rot90(full, -1))
            if full.shape != (h, w):
                full = np.asarray(Image.fromarray(full).resize((w, h), Image.Resampling.BILINEAR))
        return np.clip(full, 0, 1).astype(np.float32), prob[0, 0].numpy()
