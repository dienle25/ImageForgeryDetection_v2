import json
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

from conftest import encode

ROOT = Path(__file__).resolve().parents[1]


def run(args, cwd=ROOT, config=None):
    extra = ["--config", str(config)] if config is not None and "--config" not in map(str, args) else []
    return subprocess.run([sys.executable, "-m", "ifd", *map(str, args), *extra], cwd=cwd, capture_output=True,
                          text=True, timeout=600, encoding="utf-8")


def test_folder_run_with_corrupt_file_and_history(tmp_path, light_config):
    src = tmp_path / "in"
    (src / "sub").mkdir(parents=True)
    rng = np.random.default_rng(1)
    (src / "a.png").write_bytes(encode(rng.integers(0, 255, (90, 130, 3), dtype=np.uint8)))
    (src / "sub" / "b.jpg").write_bytes(encode(rng.integers(0, 255, (70, 70, 3), dtype=np.uint8), "JPEG"))
    (src / "broken.jpg").write_bytes(b"not a jpeg")
    (src / "notes.txt").write_text("ignored")
    out = tmp_path / "out"
    r1 = run([src, "--output", out, "--recursive"], config=light_config)
    assert r1.returncode == 1, r1.stderr  # partial failure
    latest = json.loads((out / "latest_run.json").read_text())
    rep = json.loads((out / latest["run_directory"] / "results.json").read_text())
    assert rep["counts"] == {"ok": 2, "error": 1}
    by = {r["input"]: r for r in rep["results"]}
    assert by["broken.jpg"]["status"] == "error" and by["broken.jpg"]["output_dir"] is None
    ok_dir = out / by["sub/b.jpg"]["output_dir"]
    files = {p.name for p in ok_dir.iterdir()}
    assert {"result.json", "overlay.png", "heatmap.png", "mask.png", "original.png", "heatmap.npy"} <= files
    res = json.loads((ok_dir / "result.json").read_text())
    assert res["image"]["display_size_wh"] == [70, 70]
    with Image.open(ok_dir / "overlay.png") as im:
        assert im.size == (70, 70)
    csv_text = (out / latest["run_directory"] / "results.csv").read_bytes()
    assert csv_text.startswith(b"\xef\xbb\xbf")  # UTF-8 BOM for Excel
    # second run: the earlier run directory is preserved and a new run id is created
    (src / "a.png").write_bytes(b"corrupted now")
    r2 = run([src, "--output", out, "--recursive"], config=light_config)
    assert r2.returncode == 1
    latest2 = json.loads((out / "latest_run.json").read_text())
    assert latest2["run_id"] != latest["run_id"]
    assert (out / latest["run_directory"] / "results.json").exists()
    rep2 = json.loads((out / latest2["run_directory"] / "results.json").read_text())
    assert {r["input"]: r["status"] for r in rep2["results"]}["a.png"] == "error"


def test_single_file_and_bad_paths(tmp_path, light_config):
    img = tmp_path / "x.png"
    img.write_bytes(encode(np.full((64, 64, 3), 120, np.uint8)))
    r = run([img, "--output", tmp_path / "o"], config=light_config)
    assert r.returncode == 0, r.stderr
    r = run([tmp_path / "missing.png", "--output", tmp_path / "o"], config=light_config)
    assert r.returncode == 2 and "không tồn tại" in r.stderr
    empty = tmp_path / "empty"
    empty.mkdir()
    r = run([empty, "--output", tmp_path / "o2"], config=light_config)
    assert r.returncode == 2 and "Không tìm thấy" in r.stderr
    r = run([empty, "--output", empty], config=light_config)
    assert r.returncode == 2


def test_output_inside_input_not_rescanned(tmp_path, light_config):
    src = tmp_path / "in"
    src.mkdir()
    (src / "a.png").write_bytes(encode(np.full((40, 40, 3), 90, np.uint8)))
    out = src / "results"
    assert run([src, "--output", out, "--recursive"], config=light_config).returncode == 0
    r = run([src, "--output", out, "--recursive"], config=light_config)
    assert r.returncode == 0
    latest = json.loads((out / "latest_run.json").read_text())
    rep = json.loads((out / latest["run_directory"] / "results.json").read_text())
    assert [x["input"] for x in rep["results"]] == ["a.png"], "outputs must never be re-analysed as inputs"


def test_bad_config_exit_code(tmp_path):
    cfg = tmp_path / "config.json"
    data = json.loads((ROOT / "config.json").read_text())
    data["detectors"]["mesorch"]["thresholds"]["low"] = 0.9
    data["detectors"]["mesorch"]["thresholds"]["high"] = 0.1
    cfg.write_text(json.dumps(data))
    img = tmp_path / "x.png"
    img.write_bytes(encode(np.full((32, 32, 3), 1, np.uint8)))
    r = run([img, "--output", tmp_path / "o", "--config", cfg])
    assert r.returncode == 2 and "low" in r.stderr


def test_csv_has_one_column_pair_per_detector(tmp_path, light_config):
    img = tmp_path / "x.jpg"
    img.write_bytes(encode(np.full((50, 60, 3), 30, np.uint8), "JPEG"))
    out = tmp_path / "o"
    assert run([img, "--output", out], config=light_config).returncode == 0
    latest = json.loads((out / "latest_run.json").read_text(encoding="utf-8"))
    header = (out / latest["run_directory"] / "results.csv").read_text(encoding="utf-8-sig").splitlines()[0].split(",")
    for col in ("overall_decision", "pscc_net_score", "pscc_net_decision", "commfor_aigen_score", "commfor_aigen_decision"):
        assert col in header, header


def test_no_images_writes_only_result_json(tmp_path, light_config):
    src = tmp_path / "in"
    src.mkdir()
    (src / "a.png").write_bytes(encode(np.full((80, 120, 3), 90, np.uint8)))
    out = tmp_path / "out"
    r = run([src, "--output", out, "--no-images"], config=light_config)
    assert r.returncode == 0, r.stderr
    latest = json.loads((out / "latest_run.json").read_text())
    rep = json.loads((out / latest["run_directory"] / "results.json").read_text())
    folder = out / rep["results"][0]["output_dir"]
    assert sorted(p.name for p in folder.iterdir()) == ["result.json"]
    res = json.loads((folder / "result.json").read_text())
    assert res["artifacts"] == ["result.json"] and res["summary"]["overall_decision"]
