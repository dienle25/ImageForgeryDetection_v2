"""DefakerOne2 as main model + Mesorch heatmap for reference: config, thresholds read from the config (calibrated band ->
"Không kết luận", otherwise "chưa hiệu chỉnh" 0.5), the "Hai mô hình không thống nhất" line (never a merged score),
the web page layout and the validation kit.  A fake local server replaces the DefakerOne2 model."""
import contextlib
import copy
import csv
import io
import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from conftest import encode, light_config_data
from fake_defaker2 import ANSWER, dead_url, fake_server
from ifd.analysis import Analyzer, decision_config_hash, disagreeing, report_files, summarize
from ifd.config import Config, load_config
from ifd.detectors.base import DetectorResult, decide
from ifd.detectors.defaker2 import threshold_info

ROOT = Path(__file__).resolve().parents[1]
MESORCH_CFG = ROOT / "research" / "configs" / "config_defaker2_mesorch.json"
CALIBRATED = {"low": 0.3, "high": 0.8, "status": "test",
              "calibration": {"date": "2026-11-08", "experiment_id": "D2T00_test", "proposal_file": "x.json"}}


def test_mesorch_config_file_matches_the_request():
    cfg = load_config(MESORCH_CFG)
    std = load_config(ROOT / "config.json")
    enabled = [k for k, d in cfg.data["detectors"].items() if d.get("enabled")]
    assert enabled == ["defaker2", "mesorch"]                                       # commfor / face_clip off (RAM)
    assert cfg.detector("defaker2")["role"] == "primary"
    assert not cfg.detector("commfor_aigen")["enabled"] and not cfg.detector("face_clip")["enabled"]
    mes, locked = cfg.detector("mesorch"), std.detector("mesorch")
    assert mes["thresholds"] == locked["thresholds"] and mes["image_score"] == locked["image_score"]   # locked thresholds
    assert cfg.path(mes["model"]).resolve() == std.path(locked["model"]).resolve()
    assert mes.get("role") is None                                                   # Mesorch never decides
    info = threshold_info(cfg.detector("defaker2"))
    assert info["status"] == "author_rule" and not info["calibrated"] and "Chưa hiệu chỉnh" in info["text_vi"]
    assert "không thống nhất" in cfg.notice_vi and "Mesorch" in cfg.notice_vi
    raw = MESORCH_CFG.read_bytes()
    assert b"\r\n" in raw                                                            # same line endings as config_defaker2_chinh.json
    bat = (ROOT / "RUN_APP_DEFAKER2_MESORCH.bat").read_bytes()
    assert rb"research\configs\config_defaker2_mesorch.json" in bat and b"\r\n" in bat and b'call "%~dp0RUN_APP.bat"' in bat
    bat.decode("ascii")


def test_thresholds_are_read_from_the_config_and_band_is_inconclusive():
    data = light_config_data()
    for det in data["detectors"].values():
        det["enabled"] = False
    img = encode(np.random.default_rng(1).integers(0, 255, (64, 96, 3), dtype=np.uint8))
    results = {}
    for p in (0.9, 0.5, 0.1):
        with fake_server(lambda body, p=p: (200, {**ANSWER, "p_fake": p, "p_real": 1 - p})) as (url, _):
            data["detectors"]["defaker2"] = {"enabled": True, "type": "defaker2_http", "url": url, "role": "primary",
                                             "timeout_seconds": 10, "thresholds": copy.deepcopy(CALIBRATED)}
            results[p] = Analyzer(Config(data, ROOT)).analyze(img, name="x.png")
    r = results[0.5].results[0]
    assert r.decision == "inconclusive" and r.to_json()["decision_vi"] == "Không kết luận"
    s = results[0.5].summary
    assert s["overall_decision"] == "inconclusive" and s["headline_vi"].startswith("Không kết luận")
    assert "mô hình chính" in s["basis_vi"] and "đã hiệu chỉnh" in s["basis_vi"]
    assert results[0.9].summary["overall_decision"] == "suspicious"
    assert results[0.1].summary["overall_decision"] == "no_clear_sign"
    assert r.details["threshold_info"]["calibrated"] is True and r.details["validation"]["status"] == "calibrated"
    assert r.thresholds["low"] == 0.3 and r.thresholds["high"] == 0.8
    assert "Không kết luận" in r.details["threshold_info"]["text_vi"] and "0,8" in r.details["threshold_info"]["text_vi"]


def test_uncalibrated_config_keeps_the_author_rule():
    for path in (MESORCH_CFG, ROOT / "research" / "configs" / "config_defaker2_chinh.json"):
        det = load_config(path).detector("defaker2")
        info = threshold_info(det)
        assert (info["low"], info["high"], info["band"], info["calibrated"]) == (0.5, 0.5, False, False)
        assert info["text_vi"].startswith("Chưa hiệu chỉnh")
    manual = threshold_info({"thresholds": {"low": 0.4, "high": 0.9}})       # hand-written, no calibration block
    assert manual["status"] == "uncalibrated" and "Chưa hiệu chỉnh" in manual["text_vi"] and manual["band"]


def test_decision_hash_ignores_the_calibration_block_but_not_the_values():
    cfg = load_config(MESORCH_CFG)
    h = decision_config_hash(cfg)
    data = copy.deepcopy(cfg.data)
    data["detectors"]["defaker2"]["thresholds"]["calibration"] = {"date": "2026-11-08"}
    assert decision_config_hash(Config(data, cfg.base_dir)) == h
    data["detectors"]["defaker2"]["thresholds"]["high"] = 0.9
    assert decision_config_hash(Config(data, cfg.base_dir)) != h


def _res(key, decision, primary=False, score=0.5):
    return DetectorResult(key=key, task="whole_image_authenticity" if primary else "local_manipulation", score=score,
                          score_type="x", calibrated=False, decision=decision, thresholds={}, primary=primary,
                          details={"threshold_info": {"status": "author_rule"}} if primary else {})


@pytest.mark.parametrize("d2, mes, expected", [
    ("suspicious", "no_clear_sign", True), ("no_clear_sign", "suspicious", True),
    ("suspicious", "suspicious", False), ("suspicious", "needs_review", False),
    ("inconclusive", "suspicious", False), ("no_clear_sign", "needs_review", False),
])
def test_disagreement_line_without_merging_scores(d2, mes, expected):
    cfg = load_config(MESORCH_CFG)
    results = [_res("defaker2", d2, primary=True, score=0.77), _res("mesorch", mes, score=0.99)]
    s = summarize(results, cfg)
    assert s["overall_decision"] == d2 and s["decided_by"] == "defaker2"          # the main model alone decides
    assert [r.score for r in results] == [0.77, 0.99]                             # nothing merged or rescaled
    assert bool(disagreeing("defaker2", {"defaker2": d2, "mesorch": mes})) is expected
    if expected:
        assert s["disagreement"]["text_vi"] == "Hai mô hình không thống nhất, cần xem kỹ"
        assert s["disagreement"]["detectors"] == ["defaker2", "mesorch"]
        assert "Không gộp điểm" in s["disagreement"]["detail_vi"] and "Mesorch" in s["disagreement"]["detail_vi"]
    else:
        assert "disagreement" not in s


def test_no_disagreement_line_when_the_main_model_is_unavailable():
    cfg = load_config(MESORCH_CFG)
    s = summarize([_res("defaker2", "unavailable", primary=True, score=None), _res("mesorch", "suspicious")], cfg)
    assert s["decided_by"] == "fallback" and "disagreement" not in s


@pytest.fixture(scope="module")
def mesorch_app(tmp_path_factory):
    """DefakerOne2 (fake server, answers 'Fake' 0.93) + the real Mesorch ONNX model, loaded once for the module."""
    tmp = tmp_path_factory.mktemp("mesorch_cfg")
    with fake_server() as (url, received):
        data = json.loads(MESORCH_CFG.read_text(encoding="utf-8"))
        data["detectors"]["defaker2"]["url"] = url
        data["detectors"]["mesorch"]["model"] = str((ROOT / "models" / "mesorch.onnx").resolve())
        path = tmp / "config_defaker2_mesorch.json"
        path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        cfg = load_config(path)
        yield path, cfg, Analyzer(cfg), received


def test_end_to_end_main_model_with_mesorch_heatmap(mesorch_app):
    path, cfg, an, received = mesorch_app
    rgb = np.random.default_rng(4).integers(0, 255, (300, 400, 3), dtype=np.uint8)
    rep = an.analyze(encode(rgb), name="Screenshot 2026-10-18 181055.png")
    keys = [r.key for r in rep.results]
    assert keys == ["defaker2", "mesorch"] and rep.primary_heatmap_key == "mesorch"
    d2, mes = rep.results
    th = cfg.detector("mesorch")["thresholds"]
    assert mes.decision == decide(mes.score, th["low"], th["high"]) and mes.heatmap.shape == (300, 400)
    s = rep.summary
    assert s["decided_by"] == "defaker2" and s["overall_decision"] == d2.decision == "suspicious"
    assert ("disagreement" in s) == bool(disagreeing("defaker2", {"defaker2": d2.decision, "mesorch": mes.decision}))
    files = {}
    data = json.loads(report_files(rep, cfg.overlay_alpha, files)["result.json"])
    assert data["visualization"]["heatmap_detector"] == "mesorch" and "overlay" in files
    assert data["input_quality"]["low_confidence"] is True                        # 0,12 MP + screenshot name
    assert [c["code"] for c in data["input_quality"]["reasons"]] == ["small_image", "screenshot_name"]
    cards = {c["detector"]: c for c in data["detectors"]}
    assert cards["defaker2"]["primary"] is True and "primary" not in cards["mesorch"]
    from ifd.webapp.server import State
    st = State(path, analyzer=an).status()
    items = {d["key"]: d for d in st["detectors"]}
    assert items["defaker2"]["primary"] is True and items["defaker2"]["threshold_info"]["status"] == "author_rule"
    assert items["mesorch"].get("reference") is True and "không thống nhất" in st["notice_vi"]
    assert len(received) == 1


def test_web_page_puts_the_main_card_first_and_the_heatmap_under_it():
    js = (ROOT / "ifd" / "webapp" / "static" / "app.js").read_text(encoding="utf-8")
    html = (ROOT / "ifd" / "webapp" / "static" / "index.html").read_text(encoding="utf-8")
    css = (ROOT / "ifd" / "webapp" / "static" / "style.css").read_text(encoding="utf-8")
    for elem in ('id="primary-slot"', 'id="viewer"', 'id="viewer-title"', 'id="verdict-quality"', 'id="verdict-disagree"',
                 'id="downloads"'):
        assert elem in html
    assert html.index('id="primary-slot"') < html.index('id="detectors"') < html.index('id="viewer"')
    assert '"Vùng nghi sửa (" + (NAMES[heatKey] || heatKey) + "), tham khảo"' in js and 'mesorch: "Mesorch"' in js
    assert 'insertBefore($("viewer"), heatUnderMain ? $("detectors") : $("downloads"))' in js
    assert "rep.input_quality" in js and "rep.summary.disagreement" in js and "CHƯA HIỆU CHỈNH" in js
    assert ".chip.inconclusive" in css and ".verdict.inconclusive" in css
    node = shutil.which("node")
    if node:                                                                       # syntax check when Node.js exists
        r = subprocess.run([node, "-e", "new Function(require('fs').readFileSync(process.argv[1],'utf8'))",
                            str(ROOT / "ifd" / "webapp" / "static" / "app.js")], capture_output=True, text=True)
        assert r.returncode == 0, r.stderr


def test_validation_kit_with_calibrated_band_and_disagreement_column(tmp_path):
    from ifd.company_eval import main as kit_main
    d = tmp_path / "d"
    d.mkdir()
    sizes = {"a.png": (64, 80), "b.png": (64, 96), "c.png": (64, 112), "e.png": (64, 128)}
    p_by_width = {80: 0.1, 96: 0.5, 112: 0.9, 128: 0.95}
    rng = np.random.default_rng(2)
    for name, (h, w) in sizes.items():
        (d / name).write_bytes(encode(rng.integers(0, 255, (h, w, 3), dtype=np.uint8)))
    (tmp_path / "nhan.csv").write_text("file,label,loai,nhom\na.png,0,that,g1\nb.png,0,that,g2\nc.png,1,sua,g3\ne.png,1,sua,g4\n",
                                       encoding="utf-8")

    def respond(body):
        w = Image.open(io.BytesIO(body)).size[0]
        return 200, {**ANSWER, "p_fake": p_by_width[w], "p_real": 1 - p_by_width[w]}

    with fake_server(respond) as (url, _):
        data = light_config_data()                                                 # commfor (+pscc) as reference
        data["detectors"]["defaker2"] = {"enabled": True, "type": "defaker2_http", "url": url, "role": "primary",
                                         "timeout_seconds": 10, "thresholds": copy.deepcopy(CALIBRATED)}
        cfg = tmp_path / "cfg.json"
        cfg.write_text(json.dumps(data), encoding="utf-8")
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            rc = kit_main([str(d), "--labels", str(tmp_path / "nhan.csv"), "--output", str(tmp_path / "out"),
                           "--config", str(cfg)])
    assert rc == 0
    work = next((tmp_path / "out" / "kiem_dinh").iterdir())
    rep = next(p for p in work.iterdir() if p.is_dir())
    rows = {r["file"]: r for r in csv.DictReader(io.StringIO((rep / "du_doan.csv").read_text(encoding="utf-8-sig")))}
    assert rows["b.png"]["overall_decision"] == "inconclusive" and rows["b.png"]["overall_decision_vi"] == "Không kết luận"
    assert rows["a.png"]["overall_decision"] == "no_clear_sign" and rows["c.png"]["overall_decision"] == "suspicious"
    assert all(r["mo_hinh_khong_thong_nhat"] in ("0", "1") for r in rows.values())
    m = json.loads((rep / "metrics.json").read_text(encoding="utf-8"))
    assert m["primary_detector"] == "defaker2" and m["disagreement"]["n"] == 4
    assert m["product"]["tu_can_kiem_tra_tro_len_la_duong"]["fp"] == 1                   # b.png (0,5) is in the band
    assert m["product"]["nghi_van_la_duong"]["fp"] == 0
    md = (rep / "bao_cao_kiem_dinh.md").read_text(encoding="utf-8")
    assert "“Không kết luận” trở lên" in md and "low (“Không kết luận”)" in md and "Ngưỡng defaker2: Ngưỡng đã hiệu chỉnh" in md


def test_server_down_with_mesorch_config_gives_no_conclusion_from_defaker2():
    data = json.loads(MESORCH_CFG.read_text(encoding="utf-8"))
    data["detectors"]["defaker2"]["url"] = dead_url()
    data["detectors"]["mesorch"]["enabled"] = False                                 # keep this check light
    rep = Analyzer(Config(data, MESORCH_CFG.parent)).analyze(encode(np.full((64, 64, 3), 90, np.uint8)), name="x.png")
    assert rep.summary["overall_decision"] == "unavailable" and "disagreement" not in rep.summary
