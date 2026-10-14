"""Record reference outputs of every shipped ONNX model on fixed pseudo-random inputs (tests/data/model_probes.json).

The self-test (`python -m ifd.selftest`, SELF_TEST.bat) re-runs the same inputs on the user's machine and compares,
which catches corrupted/mis-assembled model files, wrong runtime versions and platform numeric drift.
References are ONNX Runtime CPU outputs on the build machine; each ONNX file was itself checked against PyTorch at
export time (see models/manifest.json "conversion").

    python tools/make_probe_refs.py
"""
from __future__ import annotations

import json
import platform
from pathlib import Path

import numpy as np
import onnxruntime as ort

ROOT = Path(__file__).resolve().parents[1]
PROBES = {
    "mesorch.onnx": {"input_name": "image", "input_shape": [1, 3, 512, 512], "seed": 101, "tolerance": 2e-3},
    "commfor_vits16_384.onnx": {"input_name": "image", "input_shape": [1, 3, 384, 384], "seed": 102, "tolerance": 2e-2},
    "face_clip_vitl14.onnx": {"input_name": "face", "input_shape": [2, 3, 224, 224], "seed": 103, "tolerance": 2e-2},
    "pscc_net.onnx": {"input_name": "image", "input_shape": [1, 3, 256, 256], "seed": 104, "tolerance": 2e-3},
}


def main():
    out = {"generated_with": {"onnxruntime": ort.__version__, "numpy": np.__version__, "platform": platform.platform()},
           "note": "inputs: numpy.random.default_rng(seed).normal(size=input_shape).astype(float32); "
                   "first output; 256 fixed sample positions + global mean",
           "models": {}}
    for name, p in PROBES.items():
        so = ort.SessionOptions()
        so.intra_op_num_threads = 1
        sess = ort.InferenceSession(str(ROOT / "models" / name), so, providers=["CPUExecutionProvider"])
        x = np.random.default_rng(p["seed"]).normal(size=tuple(p["input_shape"])).astype(np.float32)
        y = sess.run(None, {p["input_name"]: x})[0].astype(np.float64).reshape(-1)
        idx = np.unique(np.random.default_rng(p["seed"] + 1).integers(0, y.size, size=256))
        out["models"][name] = {**p, "output_size": int(y.size), "mean": float(y.mean()),
                               "sample_index": idx.tolist(), "sample_values": y[idx].tolist()}
        print(name, y.size, float(y.mean()))
        del sess
    (ROOT / "tests" / "data" / "model_probes.json").write_text(json.dumps(out) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
