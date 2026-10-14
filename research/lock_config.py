"""Write development-derived thresholds into config.json (standard) and research/configs/config_strict.json, then
print the fingerprint to be logged as P04 BEFORE any final-test image is scored.

  python research/lock_config.py --standard thr_standard.json --strict thr_strict.json
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def apply(cfg: dict, thr: dict, label: str, rule: str) -> dict:
    out = copy.deepcopy(cfg)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    for key, info in thr["detectors"].items():
        if key not in out["detectors"] or not out["detectors"][key].get("enabled"):
            continue
        th = out["detectors"][key]["thresholds"]
        th["low"] = round(float(info["low"]), 6)
        th["high"] = round(float(info["high"]), 6)
        if th["low"] > th["high"]:
            th["low"] = th["high"]
        th["status"] = (f"{label}: chọn trên DEV ngày {stamp} theo quy tắc {rule} trên TỪNG nguồn ảnh thật của dev "
                        f"(research/derive_thresholds.py); chưa hiệu chỉnh xác suất; chưa kiểm chứng trên dữ liệu công ty")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--standard", type=Path, required=True)
    ap.add_argument("--strict", type=Path, required=True)
    a = ap.parse_args()
    cfg = json.loads((ROOT / "config.json").read_text(encoding="utf-8"))
    std = apply(cfg, json.loads(a.standard.read_text(encoding="utf-8")), "Ngưỡng chuẩn", "high: FPR≤5 %, low: FPR≤20 %")
    strict = apply(cfg, json.loads(a.strict.read_text(encoding="utf-8")), "Ngưỡng nghiêm (tùy chọn)", "high: FPR≤1 %, low: FPR≤5 %")
    for k, d in strict["detectors"].items():
        for f in ("model", "face_model"):
            if f in d:
                d[f] = "../../" + d[f]          # config_strict.json lives in research/configs/
    (ROOT / "config.json").write_text(json.dumps(std, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    (ROOT / "research" / "configs" / "config_strict.json").write_text(json.dumps(strict, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    from ifd.config import load_config
    load_config(ROOT / "config.json")                                   # validates
    load_config(ROOT / "research" / "configs" / "config_strict.json")
    sys.path.insert(0, str(ROOT / "tools"))
    from app_fingerprint import fingerprint
    fp = fingerprint(ROOT)
    import hashlib
    fp["config_strict_sha256"] = hashlib.sha256((ROOT / "research" / "configs" / "config_strict.json").read_bytes()).hexdigest()
    print(json.dumps({"standard": {k: v["thresholds"] for k, v in std["detectors"].items() if v.get("enabled")},
                      "strict": {k: v["thresholds"] for k, v in strict["detectors"].items() if v.get("enabled")},
                      "fingerprint": fp}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
