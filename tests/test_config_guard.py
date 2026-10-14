import copy
import json

import numpy as np

from conftest import encode
from ifd.analysis import CONFIG_CHANGED_VI, Analyzer, decision_config_hash
from ifd.config import Config, load_config


def test_decision_hash_ignores_status_paths_threads_but_not_thresholds(light_config):
    cfg = load_config(light_config)
    h = decision_config_hash(cfg)
    data = copy.deepcopy(cfg.data)
    data["runtime"]["threads"] = 3
    for det in data["detectors"].values():
        det["thresholds"]["status"] = "đã sửa chữ"
        det["model"] = det["model"].replace("models", "models")
    assert decision_config_hash(Config(data, cfg.base_dir)) == h
    data["detectors"]["commfor_aigen"]["thresholds"]["high"] = 0.5
    assert decision_config_hash(Config(data, cfg.base_dir)) != h
    data2 = copy.deepcopy(cfg.data)
    data2["detectors"]["pscc_net"]["enabled"] = False
    assert decision_config_hash(Config(data2, cfg.base_dir)) != h


def test_results_warn_when_config_differs_from_evaluated(tmp_path, light_config, monkeypatch):
    import ifd.analysis as analysis
    monkeypatch.setattr(analysis, "APP_ROOT", tmp_path / "no_app_root")      # isolate from the shipped summary
    cfg = load_config(light_config)
    rep_dir = cfg.base_dir / "reports"
    rep_dir.mkdir(exist_ok=True)
    img = encode(np.full((64, 80, 3), 128, np.uint8))
    summary = rep_dir / "validation_summary_vi.json"
    summary.write_text(json.dumps({"evaluated_decision_configs": [{"name": "chuẩn", "sha256": "0" * 64}]}), encoding="utf-8")
    an = Analyzer(cfg)
    assert an.config_matches_evaluated() is False and an.evaluated_profile() is None
    r = an.analyze(img, name="x.png")
    assert r.summary["warnings"][0] == CONFIG_CHANGED_VI and r.metadata["config_matches_evaluated"] is False
    summary.write_text(json.dumps({"evaluated_decision_configs": [{"name": "chuẩn", "sha256": "0" * 64},
                                                                  {"name": "nghiêm", "sha256": decision_config_hash(cfg)}]}),
                       encoding="utf-8")
    an = Analyzer(cfg)
    r = an.analyze(img, name="x.png")
    assert CONFIG_CHANGED_VI not in r.summary["warnings"] and r.metadata["config_matches_evaluated"] is True
    assert r.metadata["evaluated_profile"] == "nghiêm"
    summary.write_text(json.dumps({"evaluated_decision_config_sha256": decision_config_hash(cfg)}), encoding="utf-8")
    assert Analyzer(cfg).config_matches_evaluated() is True                 # legacy single-hash field
    summary.unlink()
    assert Analyzer(cfg).config_matches_evaluated() is None


def test_summary_found_in_app_root_for_configs_elsewhere(tmp_path, light_config, monkeypatch):
    import ifd.analysis as analysis
    root = tmp_path / "app"
    (root / "reports").mkdir(parents=True)
    (root / "reports" / "validation_summary_vi.json").write_text(
        json.dumps({"detectors": {"commfor_aigen": {"status": "experimental", "text_vi": "x"}}}), encoding="utf-8")
    monkeypatch.setattr(analysis, "APP_ROOT", root)
    cfg = load_config(light_config)                                          # lives in tmp_path, no reports/ there
    assert analysis.load_validation_summary(cfg.base_dir)["commfor_aigen"]["status"] == "experimental"
