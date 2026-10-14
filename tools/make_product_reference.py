"""Record the shipped pipeline's scores on tests/data/*.png (tests/data/product_reference.json).

The self-test compares the user's machine against these numbers (default tolerance 2e-3), which checks that the
Windows runtime (ONNX Runtime, OpenCV, Pillow, NumPy builds) reproduces the Linux build machine end-to-end,
including decoding and resizing — not only the raw networks (that is tests/data/model_probes.json).

    python tools/make_product_reference.py
"""
from __future__ import annotations

import json
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    from ifd.analysis import Analyzer, config_hash
    from ifd.config import load_config
    import onnxruntime
    cfg = load_config(ROOT / "config.json")
    an = Analyzer(cfg)
    out = {"generated_on": platform.platform(), "onnxruntime": onnxruntime.__version__, "config_sha256": config_hash(cfg),
           "tolerance": 2e-3, "images": {}}
    for p in sorted((ROOT / "tests" / "data").glob("*.png")):
        rep = an.analyze(p)
        out["images"][p.name] = {"scores": {r.key: r.score for r in rep.results}, "overall": rep.summary["overall_decision"]}
        print(p.name, out["images"][p.name])
    (ROOT / "tests" / "data" / "product_reference.json").write_text(json.dumps(out, indent=1) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
