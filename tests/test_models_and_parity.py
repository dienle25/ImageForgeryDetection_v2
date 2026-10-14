"""Model integrity, numerical parity and detector contracts (no torch required)."""
import hashlib
import json
from pathlib import Path

import numpy as np
import onnxruntime as ort
import pytest
from PIL import Image

from ifd.errors import ForgeryError
from ifd.model_store import verify_model

ROOT = Path(__file__).resolve().parents[1]
REF = ROOT / "tests" / "data" / "pscc_torch_reference.json"
PROBES = ROOT / "tests" / "data" / "model_probes.json"


def test_manifest_hashes_match_files():
    man = json.loads((ROOT / "models" / "manifest.json").read_text())
    assert {"mesorch.onnx", "commfor_vits16_384.onnx", "face_clip_vitl14.onnx",
            "face_detection_yunet_2023mar.onnx", "pscc_net.onnx"} <= set(man["files"])
    for name, entry in man["files"].items():
        p = ROOT / "models" / name
        assert p.stat().st_size == entry["size_bytes"], name
        h = hashlib.sha256()
        with p.open("rb") as f:
            for chunk in iter(lambda: f.read(1 << 22), b""):
                h.update(chunk)
        assert h.hexdigest() == entry["sha256"], name
        for k in ("source", "license_code", "license_weights", "task"):
            assert entry.get(k), (name, k)


def test_corrupt_or_unknown_model_rejected(tmp_path):
    man = ROOT / "models" / "manifest.json"
    good = ROOT / "models" / "pscc_net.onnx"
    bad = tmp_path / "pscc_net.onnx"
    data = bytearray(good.read_bytes())
    data[1000] ^= 0xFF
    bad.write_bytes(bytes(data))
    with pytest.raises(ForgeryError, match="SHA-256"):
        verify_model(bad, man)
    bad.write_bytes(b"short")
    with pytest.raises(ForgeryError, match="dung lượng"):
        verify_model(bad, man)
    with pytest.raises(ForgeryError, match="không có trong"):
        verify_model(tmp_path / "unknown.onnx", man)
    with pytest.raises(ForgeryError, match="Thiếu"):
        verify_model(tmp_path / "missing" / "pscc_net.onnx", man)


@pytest.fixture(scope="module")
def pscc():
    from ifd.detectors.pscc import PSCCDetector
    th = {"low": 0.35, "high": 0.65, "pixel": 0.5}
    return PSCCDetector(ROOT / "models" / "pscc_net.onnx", {"policy": {"mode": "bounded_native", "max_side": 512}, "thresholds": th})


def test_pscc_onnx_matches_stored_torch_reference(pscc):
    """Reference scores were produced by the original PyTorch PSCC-Net (torch 2.6.0, CPU) on fixed inputs."""
    ref = json.loads(REF.read_text())
    for case in ref["cases"]:
        rng = np.random.default_rng(case["seed"])
        x = rng.random((3, case["h"], case["w"]), dtype=np.float32)
        out = pscc.session.run(None, {"image": x[None]})
        lg = out[1][0].astype(np.float64)
        e = np.exp(lg - lg.max())
        assert abs(e[1] / e.sum() - case["score"]) < 1e-5, case
        assert abs(float(out[0].mean()) - case["mask_mean"]) < 1e-5, case


def test_pscc_author_sample_scores(pscc):
    ref = json.loads(REF.read_text())
    for name, score in ref["author_samples_native"].items():
        rgb = np.asarray(Image.open(ROOT / "tests" / "data" / name).convert("RGB"))
        s, _, _ = pscc._run(rgb)
        assert abs(s - score) < 1e-5, (name, s, score)


@pytest.mark.parametrize("name", ["mesorch.onnx", "commfor_vits16_384.onnx", "face_clip_vitl14.onnx", "pscc_net.onnx"])
def test_probe_outputs_match_reference(name):
    ref = json.loads(PROBES.read_text())["models"][name]
    x = np.random.default_rng(ref["seed"]).normal(size=tuple(ref["input_shape"])).astype(np.float32)
    sess = ort.InferenceSession(str(ROOT / "models" / name), providers=["CPUExecutionProvider"])
    y = sess.run(None, {ref["input_name"]: x})[0].astype(np.float64).reshape(-1)
    got = y[np.asarray(ref["sample_index"])]
    assert np.abs(got - np.asarray(ref["sample_values"])).max() <= ref["tolerance"]
    assert abs(float(y.mean()) - ref["mean"]) <= ref["tolerance"]


def test_mesorch_score_is_max_of_fixed_scale_map(analyzer):
    det = analyzer.detectors["mesorch"]
    rgb = np.asarray(Image.open(ROOT / "tests" / "data" / "splice1.png").convert("RGB"))
    res = det.predict(rgb)
    native = det.native_map(rgb)
    assert res.score == pytest.approx(float(native.max()), abs=1e-7)
    assert res.heatmap.shape == rgb.shape[:2]
    assert 0.0 <= res.heatmap.min() and res.heatmap.max() <= 1.0
    # no per-image normalisation: a constant image must not light up just because it is the "most suspicious" pixel
    flat = det.predict(np.full((200, 300, 3), 128, np.uint8))
    assert flat.details["heatmap_stats"]["fraction_ge_pixel_threshold"] < 0.5


def test_face_detector_not_applicable_without_faces(analyzer):
    det = analyzer.detectors["face_clip"]
    res = det.predict(np.full((300, 400, 3), 90, np.uint8))
    assert res.decision == "not_applicable" and res.score is None and res.details["faces_found"] == 0


def test_face_classifier_outputs_probabilities(analyzer):
    det = analyzer.detectors["face_clip"]
    crops = [np.random.default_rng(i).integers(0, 255, (180, 150, 3), dtype=np.uint8) for i in range(2)]
    p, lg = det.p_fake(crops)
    assert p.shape == (2,) and np.all((p >= 0) & (p <= 1)) and np.isfinite(lg).all()


def test_ai_generated_detector_has_no_heatmap(analyzer):
    det = analyzer.detectors["commfor_aigen"]
    res = det.predict(np.random.default_rng(1).integers(0, 255, (400, 500, 3), dtype=np.uint8))
    assert res.heatmap is None and 0.0 <= res.score <= 1.0


def test_summary_marks_face_task_not_applicable(analyzer):
    from conftest import encode
    rep = analyzer.analyze(encode(np.full((128, 160, 3), 70, np.uint8)), name="flat.png")
    tasks_na = [t["task"] for t in rep.summary["tasks_not_applicable"]]
    assert "face_deepfake" in tasks_na
    assert {t["task"] for t in rep.summary["tasks_checked"]} == {"local_manipulation", "ai_generated"}
