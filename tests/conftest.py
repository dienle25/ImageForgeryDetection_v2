import io
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ifd.config import load_config  # noqa: E402


@pytest.fixture(scope="session")
def config():
    return load_config(ROOT / "config.json")


@pytest.fixture(scope="session")
def analyzer(config):
    from ifd.analysis import Analyzer
    return Analyzer(config)


def light_config_data():
    """Same config, but only the two small detectors enabled: used by subprocess tests of CLI mechanics
    (exit codes, run folders, artifacts), so that each subprocess does not load ~1 GB of models."""
    import copy
    import json
    data = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    data = copy.deepcopy(data)
    for key, det in data["detectors"].items():
        det["enabled"] = det["type"] in ("pscc_onnx", "commfor_onnx")
        det["model"] = str((ROOT / det["model"]).resolve())
        if "face_model" in det:
            det["face_model"] = str((ROOT / det["face_model"]).resolve())
    return data


@pytest.fixture
def light_config(tmp_path):
    import json
    p = tmp_path / "config_light.json"
    p.write_text(json.dumps(light_config_data()), encoding="utf-8")
    return p


def encode(arr, fmt="PNG", **kw):
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format=fmt, **kw)
    return buf.getvalue()


@pytest.fixture
def rgb_image():
    rng = np.random.default_rng(0)
    base = rng.integers(0, 255, (120, 200, 3), dtype=np.uint8)
    return base
