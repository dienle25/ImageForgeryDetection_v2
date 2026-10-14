"""Build an ONNX graph of PSCC-Net (eval mode, BatchNorm folded) from the official checkpoints.

Input  : image  float32 [1,3,H,W]  RGB/255 (H,W dynamic; multiples of 8 recommended)
Outputs: mask   float32 [1,1,256,256] sigmoid localization map (NLCDetection mask1)
         logits float32 [1,2]  (class 0 authentic, class 1 forged)
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from onnx_writer import Graph, node, value_info  # noqa: E402
from pth_reader import load_state_dict  # noqa: E402

EPS = 1e-5


class Builder:
    def __init__(self, g: Graph):
        self.g = g
        self.c2 = g.init(np.array([2], dtype=np.int64), "c")
        self.c4 = g.init(np.array([4], dtype=np.int64), "c")

    def conv_bn(self, x, sd, wkey, bnkey=None, bias_key=None, stride=1, pad=None, relu=False):
        w = sd[wkey].astype(np.float64)
        b = sd[bias_key].astype(np.float64) if bias_key else np.zeros(w.shape[0])
        if bnkey:
            g_, b_, m, v = (sd[bnkey + s].astype(np.float64) for s in (".weight", ".bias", ".running_mean", ".running_var"))
            scale = g_ / np.sqrt(v + EPS)
            w = w * scale[:, None, None, None]
            b = (b - m) * scale + b_
        k = w.shape[2]
        if pad is None:
            pad = k // 2
        y = self.g.op("Conv", [x, self.g.init(w.astype(np.float32)), self.g.init(b.astype(np.float32))],
                      kernel_shape=[k, k], strides=[stride, stride], pads=[pad, pad, pad, pad])
        return self.g.op("Relu", [y]) if relu else y

    def add(self, a, b):
        return self.g.op("Add", [a, b])

    def relu(self, x):
        return self.g.op("Relu", [x])

    def resize_like(self, x, target, channels):
        hw = self.g.op("Slice", [self.g.op("Shape", [target]), self.c2, self.c4])
        sizes = self.g.op("Concat", [self.g.init(np.array([1, channels], dtype=np.int64), "c"), hw], axis=0)
        return self.g.op("Resize", [x, "", "", sizes], mode="linear", coordinate_transformation_mode="align_corners")

    def resize_fixed(self, x, channels, h, w):
        sizes = self.g.init(np.array([1, channels, h, w], dtype=np.int64), "c")
        return self.g.op("Resize", [x, "", "", sizes], mode="linear", coordinate_transformation_mode="align_corners")

    def reshape(self, x, shape):
        return self.g.op("Reshape", [x, self.g.init(np.array(shape, dtype=np.int64), "c")])


def hr_bottleneck(B, x, sd, p, has_down):
    out = B.conv_bn(x, sd, p + ".conv1.weight", p + ".bn1", pad=0, relu=True)
    out = B.conv_bn(out, sd, p + ".conv2.weight", p + ".bn2", pad=1, relu=True)
    out = B.conv_bn(out, sd, p + ".conv3.weight", p + ".bn3", pad=0)
    res = B.conv_bn(x, sd, p + ".downsample.0.weight", p + ".downsample.1", pad=0) if has_down else x
    return B.relu(B.add(out, res))


def basic(B, x, sd, p):
    out = B.conv_bn(x, sd, p + ".conv1.weight", p + ".bn1", pad=1, relu=True)
    out = B.conv_bn(out, sd, p + ".conv2.weight", p + ".bn2", pad=1)
    return B.relu(B.add(out, x))


def hr_module(B, xs, sd, p, nblocks, channels):
    xs = list(xs)
    for i in range(len(xs)):
        for k in range(nblocks[i]):
            xs[i] = basic(B, xs[i], sd, f"{p}.branches.{i}.{k}")
    outs = []
    for i in range(len(xs)):
        def fuse(j, x):
            q = f"{p}.fuse_layers.{i}.{j}"
            if j > i:
                y = B.conv_bn(x, sd, q + ".0.weight", q + ".1", pad=0)
                return B.resize_like(y, xs[i], channels[i])
            y = x
            for k in range(i - j):
                y = B.conv_bn(y, sd, f"{q}.{k}.0.weight", f"{q}.{k}.1", stride=2, pad=1, relu=(k != i - j - 1))
            return y
        y = xs[0] if i == 0 else fuse(0, xs[0])
        for j in range(1, len(xs)):
            y = B.add(y, xs[j] if i == j else fuse(j, xs[j]))
        outs.append(B.relu(y))
    return outs


def hrnet(B, x, sd):
    x = B.conv_bn(x, sd, "conv1.weight", "bn1", pad=1, relu=True)
    x = B.conv_bn(x, sd, "conv2.weight", "bn2", pad=1, relu=True)
    x = hr_bottleneck(B, x, sd, "layer1.0", True)
    x = hr_bottleneck(B, x, sd, "layer1.1", False)
    xs = [B.conv_bn(x, sd, "transition1.0.0.weight", "transition1.0.1", pad=1, relu=True),
          B.conv_bn(x, sd, "transition1.1.0.0.weight", "transition1.1.0.1", stride=2, pad=1, relu=True)]
    ys = hr_module(B, xs, sd, "stage2.0", [2, 2], [18, 36])
    xs = [ys[0], ys[1], B.conv_bn(ys[-1], sd, "transition2.2.0.0.weight", "transition2.2.0.1", stride=2, pad=1, relu=True)]
    ys = hr_module(B, xs, sd, "stage3.0", [2, 2, 2], [18, 36, 72])
    xs = [ys[0], ys[1], ys[2], B.conv_bn(ys[-1], sd, "transition3.3.0.0.weight", "transition3.3.0.1", stride=2, pad=1, relu=True)]
    return hr_module(B, xs, sd, "stage4.0", [2, 2, 2, 2], [18, 36, 72, 144])


def nonlocal_mask(B, x, sd, p, c, h, w, r):
    g = B.g
    ic = c * r * r
    n = (h // r) * (w // r)
    x1 = B.reshape(x, [1, ic, h // r, w // r])
    conv = lambda name, inp: B.conv_bn(inp, sd, f"{p}.{name}.weight", None, bias_key=f"{p}.{name}.bias", pad=0)
    g_x = g.op("Transpose", [B.reshape(conv("g", x1), [1, ic, n])], perm=[0, 2, 1])   # 1,N,ic
    theta = B.reshape(conv("theta", x1), [1, ic, n])                                   # 1,mc,N
    phi = B.reshape(conv("phi", x1), [1, ic, n])                                       # 1,mc,N
    theta_t = g.op("Transpose", [theta], perm=[0, 2, 1])                               # 1,N,mc
    phi_t = g.op("Transpose", [phi], perm=[0, 2, 1])                                   # 1,N,mc
    f_s = g.op("Softmax", [g.op("MatMul", [theta_t, phi])], axis=-1)                   # 1,N,N
    f_c = g.op("Softmax", [g.op("MatMul", [theta, phi_t])], axis=-1)                   # 1,mc,mc
    y_s = B.reshape(g.op("Transpose", [g.op("MatMul", [f_s, g_x])], perm=[0, 2, 1]), [1, c, h, w])
    y_c = B.reshape(g.op("MatMul", [g_x, f_c]), [1, c, h, w])
    ws = conv("W_s", y_s)
    wc = conv("W_c", y_c)
    gs = g.init(sd[p + ".gamma_s"].reshape(1).astype(np.float32))
    gc = g.init(sd[p + ".gamma_c"].reshape(1).astype(np.float32))
    z = B.add(B.add(x, g.op("Mul", [ws, gs])), g.op("Mul", [wc, gc]))
    m = B.conv_bn(z, sd, f"{p}.getmask.0.weight", None, bias_key=f"{p}.getmask.0.bias", pad=1, relu=True)
    m = B.conv_bn(m, sd, f"{p}.getmask.2.weight", None, bias_key=f"{p}.getmask.2.bias", pad=1)
    return g.op("Sigmoid", [m])


def resized_feats(B, feats, crop=256):
    chans = [18, 36, 72, 144]
    return [B.resize_fixed(f, c, crop // s, crop // s) for f, c, s in zip(feats, chans, (1, 2, 4, 8))]


def nlc(B, feats, sd):
    g = B.g
    s1, s2, s3, s4 = resized_feats(B, feats)
    m4 = nonlocal_mask(B, s4, sd, "getmask4", 144, 32, 32, 1)
    s3 = g.op("Mul", [s3, B.resize_fixed(m4, 1, 64, 64)])
    m3 = nonlocal_mask(B, s3, sd, "getmask3", 72, 64, 64, 2)
    s2 = g.op("Mul", [s2, B.resize_fixed(m3, 1, 128, 128)])
    m2 = nonlocal_mask(B, s2, sd, "getmask2", 36, 128, 128, 2)
    s1 = g.op("Mul", [s1, B.resize_fixed(m2, 1, 256, 256)])
    return nonlocal_mask(B, s1, sd, "getmask1", 18, 256, 256, 4)


def head_bottleneck(B, x, sd, p):
    out = B.conv_bn(x, sd, p + ".conv1.weight", p + ".bn1", pad=0, relu=True)
    out = B.conv_bn(out, sd, p + ".conv2.weight", p + ".bn2", pad=1, relu=True)
    out = B.conv_bn(out, sd, p + ".conv3.weight", p + ".bn3", pad=0)
    res = B.conv_bn(x, sd, p + ".downsample.0.weight", p + ".downsample.1", pad=0)
    return B.relu(B.add(out, res))


def det_head(B, feats, sd):
    g = B.g
    ys = resized_feats(B, feats)
    y = head_bottleneck(B, ys[0], sd, "incre_modules.0.0")
    for i in range(3):
        d = B.conv_bn(y, sd, f"downsamp_modules.{i}.0.weight", f"downsamp_modules.{i}.1",
                      bias_key=f"downsamp_modules.{i}.0.bias", stride=2, pad=1, relu=True)
        y = B.add(head_bottleneck(B, ys[i + 1], sd, f"incre_modules.{i+1}.0"), d)
    y = B.conv_bn(y, sd, "final_layer.0.weight", "final_layer.1", bias_key="final_layer.0.bias", pad=0, relu=True)
    v = g.op("Flatten", [g.op("GlobalAveragePool", [y])], axis=1)
    v = g.op("Relu", [g.op("Gemm", [v, g.init(sd["classifier.0.weight"]), g.init(sd["classifier.0.bias"])], transB=1)])
    return g.op("Gemm", [v, g.init(sd["classifier.2.weight"]), g.init(sd["classifier.2.bias"])], transB=1)


def build(weights_dir: Path) -> bytes:
    sd_hr = load_state_dict(weights_dir / "HRNet.pth")
    sd_nlc = load_state_dict(weights_dir / "NLCDetection.pth")
    sd_det = load_state_dict(weights_dir / "DetectionHead.pth")
    g = Graph("pscc_net")
    B = Builder(g)
    g.inputs.append(value_info("image", 1, [1, 3, "H", "W"]))
    feats = hrnet(B, "image", sd_hr)
    mask = nlc(B, feats, sd_nlc)
    logits = det_head(B, feats, sd_det)
    g.nodes.append(node("Identity", [mask], ["mask"]))
    g.nodes.append(node("Identity", [logits], ["logits"]))
    g.outputs.append(value_info("mask", 1, [1, 1, 256, 256]))
    g.outputs.append(value_info("logits", 1, [1, 2]))
    hashes = {n: hashlib.sha256((weights_dir / n).read_bytes()).hexdigest() for n in ("HRNet.pth", "NLCDetection.pth", "DetectionHead.pth")}
    doc = ("PSCC-Net (proteus1991/PSCC-Net@53e5ff7, MIT) exported without torch by tools/pscc_onnx.py; "
           "BatchNorm folded; source checkpoint sha256: " + ", ".join(f"{k}={v}" for k, v in hashes.items()))
    return g.serialize(doc=doc)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("weights_dir", type=Path)
    ap.add_argument("output", type=Path)
    a = ap.parse_args()
    data = build(a.weights_dir)
    a.output.parent.mkdir(parents=True, exist_ok=True)
    a.output.write_bytes(data)
    print(a.output, len(data), hashlib.sha256(data).hexdigest())
