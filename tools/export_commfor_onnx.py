"""Export Community Forensics ViT-S/16@384 (Park & Owens, CVPR 2025; MIT) to ONNX and verify parity."""
import argparse, hashlib, json
import numpy as np
import torch, timm
from safetensors.torch import load_file

ap = argparse.ArgumentParser()
ap.add_argument("safetensors"); ap.add_argument("output")
a = ap.parse_args()
sd = load_file(a.safetensors)
vit = timm.create_model("vit_small_patch16_384.augreg_in21k_ft_in1k", pretrained=False)
vit.head = torch.nn.Linear(384, 1)
vit.load_state_dict({k[4:]: v for k, v in sd.items()}, strict=True)
vit.eval()
x = torch.randn(1, 3, 384, 384)
torch.onnx.export(vit, x, a.output, input_names=["image"], output_names=["logit"], opset_version=17,
                  dynamic_axes={"image": {0: "batch"}, "logit": {0: "batch"}}, dynamo=False)
import onnxruntime as ort, onnx
onnx.checker.check_model(onnx.load(a.output), full_check=True)
sess = ort.InferenceSession(a.output, providers=["CPUExecutionProvider"])
worst = 0
for seed in range(4):
    xi = torch.randn(2, 3, 384, 384, generator=torch.Generator().manual_seed(seed))
    with torch.inference_mode():
        t = vit(xi).numpy()
    o = sess.run(None, {"image": xi.numpy()})[0]
    worst = max(worst, float(np.abs(t - o).max()))
print(json.dumps({"onnx": a.output, "sha256": hashlib.sha256(open(a.output, "rb").read()).hexdigest(),
                  "max_abs_logit_diff_vs_torch": worst}))
