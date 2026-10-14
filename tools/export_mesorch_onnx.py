"""Export Mesorch (Zhu et al., AAAI 2025; MIT license) to ONNX for torch-free CPU inference.

Research-time tool (needs torch + timm + the upstream Mesorch repo); the shipped app only needs onnxruntime.

    python tools/export_mesorch_onnx.py --repo PATH/TO/Mesorch --ckpt mesorch-98.pth --out models/mesorch.onnx

Graph: input  "image"    float32 [1,3,512,512]  (RGB resized to 512x512 with cv2.INTER_LINEAR, /255, ImageNet mean/std)
       output "mask_prob" float32 [1,1,512,512]  (sigmoid of the fused logit map, fixed [0,1] scale)
The per-image min-max inside the DCT feature extractors is part of the trained architecture (input features),
NOT a normalisation of the output map.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import types
from pathlib import Path

import numpy as np
import torch
from torch import nn


def stub_imdlbenco():
    reg = types.ModuleType("IMDLBenCo.registry")

    class _Reg:
        def register_module(self, *a, **k):
            return lambda cls: cls

    reg.MODELS = _Reg()
    pkg = types.ModuleType("IMDLBenCo")
    pkg.registry = reg
    sys.modules.setdefault("IMDLBenCo", pkg)
    sys.modules.setdefault("IMDLBenCo.registry", reg)


class MesorchInference(nn.Module):
    """Same computation as MesorchFull.forward minus the training loss."""

    def __init__(self, m):
        super().__init__()
        self.m = m

    def forward(self, image):
        m = self.m
        high_freq = m.high_dct(image)
        low_freq = m.low_dct(image)
        input_high = torch.cat([image, high_freq], dim=1)
        input_low = torch.cat([image, low_freq], dim=1)
        input_all = torch.cat([image, high_freq, low_freq], dim=1)
        _, outs1 = m.convnext(input_high)
        _, outs2 = m.segformer(input_low)
        _, features = m.upsample(outs1 + outs2)
        gate = m.gate(input_all)
        reduced = torch.cat([m.inverse[i](features[i]) for i in range(8)], dim=1)
        pred = torch.sum(gate * reduced, dim=1, keepdim=True)
        pred = m.resize(pred)
        return torch.sigmoid(pred.float())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True, type=Path)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()
    stub_imdlbenco()
    sys.path.insert(0, str(a.repo))
    from mesorch import MesorchFull  # noqa

    model = MesorchFull(seg_pretrain_path=None, conv_pretrain=False)
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck)
    res = model.load_state_dict(sd, strict=True)
    model.eval()
    wrap = MesorchInference(model).eval()

    rng = np.random.default_rng(0)
    x = torch.from_numpy(rng.normal(size=(1, 3, 512, 512)).astype(np.float32))
    with torch.inference_mode():
        ref_full = model(x, torch.zeros(1, 1, 512, 512))["pred_mask"].numpy()
        ref = wrap(x).numpy()
    assert np.array_equal(ref, ref_full), "wrapper differs from MesorchFull.forward"

    a.out.parent.mkdir(parents=True, exist_ok=True)
    tmp = a.out.with_suffix(".tmp.onnx")
    torch.onnx.export(wrap, (x,), str(tmp), input_names=["image"], output_names=["mask_prob"],
                      opset_version=a.opset, do_constant_folding=True, dynamo=False)
    import onnx
    mo = onnx.load(str(tmp))
    onnx.checker.check_model(mo)
    mo.producer_name = "ifd-export-mesorch"
    mo.doc_string = ("Mesorch (Zhu et al., AAAI 2025), MIT license. Weights: official checkpoint "
                     f"sha256={hashlib.sha256(a.ckpt.read_bytes()).hexdigest()}. Output: sigmoid map 512x512.")
    onnx.save(mo, str(a.out))
    tmp.unlink()

    import onnxruntime as ort
    sess = ort.InferenceSession(str(a.out), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"image": x.numpy()})[0]
    diff = float(np.abs(got - ref).max())
    report = {"onnx": str(a.out), "sha256": hashlib.sha256(a.out.read_bytes()).hexdigest(),
              "bytes": a.out.stat().st_size, "max_abs_diff_vs_torch_random_input": diff,
              "load_missing": list(res.missing_keys), "load_unexpected": list(res.unexpected_keys),
              "opset": a.opset, "torch": torch.__version__, "onnxruntime": ort.__version__}
    print(json.dumps(report, indent=1))
    if diff > 1e-3:
        raise SystemExit("parity check failed")


if __name__ == "__main__":
    main()
