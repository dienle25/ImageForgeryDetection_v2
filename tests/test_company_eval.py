import csv
import io
import json
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

from conftest import encode

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "data"


def run(args, config):
    return subprocess.run([sys.executable, "-m", "ifd.company_eval", *map(str, args), "--config", str(config)],
                          cwd=ROOT, capture_output=True, text=True, timeout=600, encoding="utf-8")


def make_folder_dataset(base: Path) -> Path:
    data = base / "Dữ liệu công ty"
    real = data / "Thật" / "dien_thoai"
    fake = data / "Sửa" / "ghep_anh"
    real.mkdir(parents=True)
    fake.mkdir(parents=True)
    shutil.copy(DATA / "authentic1.png", real / "a1.png")
    shutil.copy(DATA / "authentic3.png", real / "a3.png")
    shutil.copy(DATA / "authentic3.png", real / "a3_copy.png")          # duplicate bytes -> counted once
    rng = np.random.default_rng(3)
    (real / "r.jpg").write_bytes(encode(rng.integers(0, 255, (80, 120, 3), dtype=np.uint8), "JPEG", quality=90))
    shutil.copy(DATA / "splice1.png", fake / "s1.png")
    shutil.copy(DATA / "copymove2.png", fake / "c2.png")
    (fake / "hong.jpg").write_bytes(b"not an image")                        # unreadable -> listed as error
    (data / "khac").mkdir()
    (data / "khac" / "x.png").write_bytes(encode(np.zeros((32, 32, 3), np.uint8)))
    (data / "le.png").write_bytes(encode(np.zeros((32, 32, 3), np.uint8)))   # stray image without label
    return data


def test_folder_mode_report_cache_and_dedup(tmp_path, light_config):
    data = make_folder_dataset(tmp_path)
    out = tmp_path / "out"
    r = run([data, "--output", out], light_config)
    assert r.returncode == 1, r.stderr                                      # one unreadable file
    assert "Đang nạp mô hình" in r.stdout
    work = next((out / "kiem_dinh").iterdir())
    reports = sorted(p for p in work.iterdir() if p.is_dir())
    assert len(reports) == 1
    m = json.loads((reports[0] / "metrics.json").read_text(encoding="utf-8"))
    assert m["counts"] == {"found": 7, "ok": 6, "errors": 1, "real": 3, "fake": 2}
    assert m["dedup"]["duplicates_counted_once"] == 1
    assert set(m["detectors"]) == {"pscc_net", "commfor_aigen"}
    assert m["detectors"]["commfor_aigen"]["n_pos"] == 2 and m["detectors"]["commfor_aigen"]["n_neg"] == 3
    assert "sua/ghep_anh" in m["product"]["by_type"] and "that/dien_thoai" in m["product"]["by_type"]
    assert any("Bỏ qua thư mục không rõ nhãn: khac" in w for w in m["warnings"])
    assert any("trực tiếp" in w for w in m["warnings"])
    assert "rows" not in m["reference_thresholds"]["commfor_aigen"]          # too few real images -> only a note
    html = (reports[0] / "bao_cao_kiem_dinh.html").read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>") and "<table>" in html and "http" not in html.split("</style>")[1]
    raw = (reports[0] / "du_doan.csv").read_bytes()
    assert raw.startswith(b"\xef\xbb\xbf")
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8-sig"))))
    assert {"pscc_net_score", "commfor_aigen_decision", "overall_decision_vi"} <= set(rows[0])
    assert sum(r["status"] == "error" for r in rows) == 1
    assert sum(r["status"] == "duplicate_or_conflict" for r in rows) == 1
    # second run: every score comes from the cache (no model load) and a NEW report folder is written
    r2 = run([data, "--output", out], light_config)
    assert r2.returncode == 1, r2.stderr
    assert "Đang nạp mô hình" not in r2.stdout
    assert len([p for p in work.iterdir() if p.is_dir()]) == 2


def test_csv_mode_conflicts_and_path_escape(tmp_path, light_config):
    data = tmp_path / "d"
    data.mkdir()
    shutil.copy(DATA / "authentic1.png", data / "x.png")
    shutil.copy(DATA / "authentic1.png", data / "x_again.png")              # same pixels, opposite label -> excluded
    shutil.copy(DATA / "splice1.png", data / "s.png")
    shutil.copy(DATA / "authentic3.png", data / "y.png")
    (tmp_path / "outside.png").write_bytes((DATA / "authentic3.png").read_bytes())
    labels = tmp_path / "nhan.csv"
    labels.write_text("file,label,loai,nhom\nx.png,0,cam,g1\nx_again.png,1,ghep,g1\ns.png,sua,ghep,g2\n"
                      "y.png,that,cam,g3\n../outside.png,0,cam,g4\nmissing_label.png,,cam,g5\n", encoding="utf-8")
    out = tmp_path / "out"
    r = run([data, "--labels", labels, "--output", out], light_config)
    assert r.returncode == 0, r.stderr
    work = next((out / "kiem_dinh").iterdir())
    m = json.loads((next(p for p in work.iterdir() if p.is_dir()) / "metrics.json").read_text(encoding="utf-8"))
    assert sorted(m["dedup"]["conflicting_labels_excluded"]) == ["x.png", "x_again.png"]
    assert m["counts"]["real"] == 1 and m["counts"]["fake"] == 1
    assert any("ra ngoài" in w for w in m["warnings"]) and any("nhãn không hợp lệ" in w for w in m["warnings"])


def test_bad_inputs_exit_2(tmp_path, light_config):
    empty = tmp_path / "e"
    empty.mkdir()
    (empty / "a.png").write_bytes(encode(np.zeros((32, 32, 3), np.uint8)))
    r = run([empty, "--output", tmp_path / "o"], light_config)
    assert r.returncode == 2 and "Không thấy thư mục nhãn" in r.stderr
    r = run([tmp_path / "khong_co", "--output", tmp_path / "o"], light_config)
    assert r.returncode == 2 and "Không tìm thấy thư mục dữ liệu" in r.stderr
    data = make_folder_dataset(tmp_path)
    r = run([data, "--output", data / "ket_qua"], light_config)
    assert r.returncode == 2 and "không được nằm trong" in r.stderr
