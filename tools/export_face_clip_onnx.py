"""Export the yermandy/deepfake-detection model (CLIP ViT-L/14 with LN-tuning + linear head; MIT) to ONNX.

Research-time tool (needs torch + onnx); the shipped app only needs onnxruntime.

  python tools/export_face_clip_onnx.py export --ts model.torchscript --out models/face_clip_vitl14.onnx

The published TorchScript hard-codes a bfloat16 cast of its input, so it cannot simply be converted to float32.
Step 1 therefore rebuilds the SAME computation as a plain PyTorch module (read from the TorchScript code):
  patch conv 14x14 (no bias) -> [CLS] + positional embedding -> pre-LN -> 24 x (x + MHSA(LN1 x); x + MLP(LN2 x),
  16 heads, scale 1/8, quick-GELU x*sigmoid(1.702x)) -> CLS token -> post-LN -> L2-normalise -> Linear(1024, 2).
  LayerNorms use the trained `ln_tuning_layers.default` parameters (eps 1e-5), exactly as the TorchScript does.
It verifies the rebuilt module against the original TorchScript (both in bf16, must be bit-identical), then exports
float32 compute with the published bf16 weights stored as BFLOAT16 initializers + Cast->float (exact, ~half size).
Input "face" float32 [N,3,224,224] (CLIP-normalised); output "logits" float32 [N,2] = [real, fake].
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

PREFIX = "feature_extractor.base_model.model.vision_model."


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def probe_input() -> np.ndarray:
    return np.random.default_rng(0).normal(size=(2, 3, 224, 224)).astype(np.float32)


def build_clean(sd):
    """Plain-PyTorch rebuild.  Parameters stay in bfloat16 (as published, shared with `sd`, no copy).
    mode["upcast"] = False -> bf16 compute (must equal the TorchScript);
    mode["upcast"] = True  -> every weight is upcast exactly to float32 at use, float32 compute (exported graph:
    BFLOAT16 initializers + Cast->FLOAT, i.e. half-size file, exact weights)."""
    import torch
    from torch import nn
    import torch.nn.functional as F

    mode = {"upcast": False}

    def W(p):
        return p.float() if mode["upcast"] else p

    def P(t):
        return nn.Parameter(t, requires_grad=False)

    class LN(nn.Module):
        def __init__(self, name):
            super().__init__()
            self.weight = P(sd[PREFIX + name + ".ln_tuning_layers.default.weight"])
            self.bias = P(sd[PREFIX + name + ".ln_tuning_layers.default.bias"])
            self.n = int(self.weight.shape[0])

        def forward(self, x):
            return F.layer_norm(x, (self.n,), W(self.weight), W(self.bias), 1e-5)

    class Lin(nn.Module):
        def __init__(self, name):
            super().__init__()
            key = PREFIX + name if (PREFIX + name + ".weight") in sd else name
            self.weight = P(sd[key + ".weight"])
            self.bias = P(sd[key + ".bias"])

        def forward(self, x):
            return F.linear(x, W(self.weight), W(self.bias))

    class Layer(nn.Module):
        def __init__(self, i):
            super().__init__()
            p = f"encoder.layers.{i}."
            self.ln1, self.ln2 = LN(p + "layer_norm1"), LN(p + "layer_norm2")
            self.q, self.k, self.v = Lin(p + "self_attn.q_proj"), Lin(p + "self_attn.k_proj"), Lin(p + "self_attn.v_proj")
            self.o = Lin(p + "self_attn.out_proj")
            self.fc1, self.fc2 = Lin(p + "mlp.fc1"), Lin(p + "mlp.fc2")

        def forward(self, x):
            b, n, d = x.shape
            h = self.ln1(x)
            q = self.q(h).view(b, n, 16, 64).transpose(1, 2)
            k = self.k(h).view(b, n, 16, 64).transpose(1, 2)
            v = self.v(h).view(b, n, 16, 64).transpose(1, 2)
            a = F.scaled_dot_product_attention(q, k, v, None, 0.0, False, scale=0.125)
            a = a.transpose(1, 2).reshape(b, n, d)
            x = x + self.o(a)
            h = self.fc1(self.ln2(x))
            h = h * torch.sigmoid(1.702 * h)
            return x + self.fc2(h)

    class CleanClipDeepfake(nn.Module):
        def __init__(self):
            super().__init__()
            self.patch = P(sd[PREFIX + "embeddings.patch_embedding.weight"])
            self.cls = P(sd[PREFIX + "embeddings.class_embedding"])
            self.pos = P(sd[PREFIX + "embeddings.position_embedding.weight"])
            self.pre = LN("pre_layrnorm")
            self.layers = nn.ModuleList([Layer(i) for i in range(24)])
            self.post = LN("post_layernorm")
            self.head = Lin("model.linear")
            self.mode = mode

        def forward(self, x):
            x = x.float() if mode["upcast"] else x.to(torch.bfloat16)
            p = F.conv2d(x, W(self.patch), stride=14).flatten(2).transpose(1, 2)
            c = W(self.cls).view(1, 1, -1).expand(p.shape[0], 1, -1)
            h = torch.cat([c, p], dim=1) + W(self.pos).unsqueeze(0)
            h = self.pre(h)
            for layer in self.layers:
                h = layer(h)
            f = self.post(h[:, 0])
            f = f / f.norm(p=2, dim=1, keepdim=True).clamp_min(1e-12)
            return self.head(f)

    return CleanClipDeepfake().eval()


def cmd_export(a):
    import gc
    import torch
    ts = torch.jit.load(str(a.ts), map_location="cpu").eval()
    sd = {k: v for k, v in ts.state_dict().items()}
    x = torch.from_numpy(probe_input())
    with torch.inference_mode():
        ref_ts_bf16 = ts(x.to(torch.bfloat16)).float().numpy()
    clean = build_clean(sd)
    with torch.inference_mode():
        clean_bf16 = clean(x).float().numpy()
        clean.mode["upcast"] = True
        ref32 = clean(x).numpy()
    src_sha = sha256(a.ts)
    del ts
    gc.collect()
    rebuild_diff = float(np.abs(clean_bf16 - ref_ts_bf16).max())
    print(json.dumps({"rebuilt_vs_torchscript_bf16_max_abs_logit": rebuild_diff,
                      "fp32_vs_torchscript_bf16_max_abs_logit": float(np.abs(ref32 - ref_ts_bf16).max())}), flush=True)
    if rebuild_diff > 1e-6:
        raise SystemExit("rebuilt module does not reproduce the TorchScript")
    # no constant folding: keeps BFLOAT16 initializers + Cast (ORT folds them in memory at session start)
    torch.onnx.export(clean, (x,), str(a.out), input_names=["face"], output_names=["logits"],
                      dynamic_axes={"face": {0: "n"}, "logits": {0: "n"}}, opset_version=a.opset,
                      do_constant_folding=False, dynamo=False)
    del clean, sd
    gc.collect()
    import onnx
    m = onnx.load(str(a.out))
    m.producer_name = "ifd-export-face-clip"
    m.doc_string = ("yermandy/deepfake-detection (MIT): CLIP ViT-L/14 LN-tuned on FF++. Weights bf16 as published "
                    f"(TorchScript sha256 {src_sha}); compute float32. Output logits [real, fake].")
    onnx.checker.check_model(m)
    onnx.save(m, str(a.out))
    del m
    gc.collect()
    import onnxruntime as ort
    sess = ort.InferenceSession(str(a.out), providers=["CPUExecutionProvider"])
    got = sess.run(None, {"face": probe_input()})[0]
    rep = {"onnx": str(a.out), "bytes": a.out.stat().st_size, "sha256": sha256(a.out),
           "rebuilt_vs_torchscript_bf16_max_abs_logit": rebuild_diff,
           "onnx_vs_torch_fp32_max_abs_logit": float(np.abs(got - ref32).max()),
           "onnx_vs_original_torchscript_bf16_max_abs_logit": float(np.abs(got - ref_ts_bf16).max()),
           "source_torchscript_sha256": src_sha, "torch": torch.__version__, "onnxruntime": ort.__version__}
    a.out.with_suffix(".export.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))
    if rep["onnx_vs_torch_fp32_max_abs_logit"] > 1e-3:
        raise SystemExit("parity check failed")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("export")
    e.add_argument("--ts", required=True, type=Path)
    e.add_argument("--out", required=True, type=Path)
    e.add_argument("--opset", type=int, default=17)
    a = ap.parse_args()
    {"export": cmd_export}[a.cmd](a)


if __name__ == "__main__":
    main()
