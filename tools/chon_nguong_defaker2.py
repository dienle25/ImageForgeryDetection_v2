"""Đề xuất ngưỡng low/high cho DefakerOne2 (mô hình chính) trên TẬP HIỆU CHỈNH RIÊNG.  NGƯỜI DÙNG tự chạy.

    tools\\CHON_NGUONG_DEFAKER2.bat                       chấm tập hiệu chỉnh (cần server DefakerOne2) rồi đề xuất
    tools\\CHON_NGUONG_DEFAKER2.bat --chi-anh-that        chỉ chấm ảnh thật (đủ để chọn ngưỡng, nhanh hơn khoảng một nửa)
    tools\\CHON_NGUONG_DEFAKER2.bat --du-doan FILE.csv    không chấm, đọc du_doan.csv đã có của tập hiệu chỉnh
    tools\\CHON_NGUONG_DEFAKER2.bat ... --ghi             ghi ngưỡng vào config DefakerOne2 + một dòng EXPERIMENT_LOG.jsonl

Tập hiệu chỉnh = file nhan_hieu_chinh.csv do tools/tao_nhan_cong_khai.py tạo: phần dev đã khóa của repo (dev_korus,
DeepFakeFace dev, OpenFake dev nếu có pyarrow).  KHÔNG BAO GIỜ dùng data.zip, A–D, CocoGlide hay test_korus: script dừng
nếu du_doan.csv có ảnh ngoài tập hiệu chỉnh hoặc trùng tập đo.

Quy tắc (giống các detector khác: QUALITY_TARGETS.yaml operating_point_policy, research/derive_thresholds.py):
  high = ngưỡng nhỏ nhất để tỷ lệ báo nhầm (FPR) <= 5 % trên TỪNG nguồn ảnh thật của tập hiệu chỉnh;
  low  = ngưỡng nhỏ nhất để FPR <= 20 % trên từng nguồn; low <= high.  Giữa hai ngưỡng app hiện "Không kết luận".
Đề xuất luôn được in và lưu vào reports\\defaker2\\nguong_de_xuat.json (+ .md).  Chỉ khi có --ghi mới ghi ngưỡng vào
config DefakerOne2 (mặc định research/configs/config_defaker2_mesorch.json; thêm --config-ghi để ghi file khác).
config.json đã khóa KHÔNG BAO GIỜ bị ghi.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import os
import shutil
import sys
import urllib.parse
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(Path(__file__).resolve().parent)):      # the bundled runtime does not add these itself
    if _p not in sys.path:
        sys.path.insert(0, _p)

import do_lai_defaker2 as dl  # noqa: E402
import tao_nhan_cong_khai as tnck  # noqa: E402
from ifd import __version__  # noqa: E402
from ifd.config import load_config  # noqa: E402
from ifd.errors import ForgeryError  # noqa: E402
from ifd_core.metrics import auroc, bootstrap_ci, threshold_for_fpr  # noqa: E402

TARGET_HIGH, TARGET_LOW = 0.05, 0.20
MIN_REAL_PER_SOURCE = 50          # below: warning (5 % of 50 = 2 false alarms allowed)
MIN_REAL_TO_WRITE = 20            # below: --ghi refused
PROPOSAL_NAME = "nguong_de_xuat.json"
LOCKED_CONFIG = ROOT / "config.json"
EXPERIMENT_LOG = ROOT / "EXPERIMENT_LOG.jsonl"
DEFAULT_WRITE = [dl.CFG_MESORCH]
SCORE_COL = "defaker2_score"
RULE_VI = ("high = ngưỡng nhỏ nhất để FPR ≤ 5 % trên TỪNG nguồn ảnh thật của tập hiệu chỉnh; low = ngưỡng nhỏ nhất để "
           "FPR ≤ 20 % trên từng nguồn; low ≤ high (như research/derive_thresholds.py và QUALITY_TARGETS.yaml). "
           "Ngưỡng làm tròn LÊN 9 chữ số thập phân. Giữa low và high app hiện “Không kết luận”.")


def sha256_file(p: Path) -> str:
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def ceil9(x: float) -> float:
    return 1.0 if x >= 1.0 else math.ceil(x * 1e9) / 1e9


def wilson(k: int, n: int, z: float = 1.96):
    if not n:
        return None
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return [max(0.0, c - h), min(1.0, c + h)]


# ------------------------------------------------------------------------------------------------- guards
def load_calibration(public_dir: Path) -> tuple[Path, list[dict], dict]:
    label = public_dir / tnck.LABEL_FILES["hieu_chinh"]
    split = public_dir / tnck.SPLIT_FILE
    if not label.is_file() or not split.is_file():
        raise dl.PrepError(f"Chưa có {label} hoặc {split}: chạy tools/tao_nhan_cong_khai.py --bo hieu_chinh trước.")
    info = json.loads(split.read_text(encoding="utf-8"))
    rec = (info.get("tap") or {}).get("hieu_chinh") or {}
    if rec.get("tach") != "hieu_chinh" or rec.get("sha256") != sha256_file(label):
        raise dl.PrepError(f"{label.name} không khớp {split.name} (file nhãn đã bị sửa?). Chạy lại tools/tao_nhan_cong_khai.py "
                           "--bo hieu_chinh.")
    return label, dl.read_labels(label), info


def test_files(public_dir: Path) -> set[str]:
    out = set()
    for bo in tnck.TEST_SETS:
        p = public_dir / tnck.LABEL_FILES[bo]
        if p.is_file():
            out |= {r["file"].replace("\\", "/") for r in dl.read_labels(p)}
    return out


def datazip_overlap(calib_rows: list[dict], data_dir: Path) -> int:
    """Calibration images (their manifest SHA-256) that are byte-identical to an image of data.zip (A–D)."""
    if not data_dir.is_dir():
        return 0
    digests = set()
    for r in tnck.load_manifest("korus_png.jsonl") + tnck.load_manifest("dff_extra.jsonl") + \
            tnck.load_manifest("dff_faceswap.jsonl") + tnck.load_manifest("openfake.jsonl"):
        if str(r.get("split", "")).startswith("dev") and r.get("sha256"):
            digests.add(r["sha256"])
    n = 0
    for sub in ("that", "sua", "doi_chung"):
        for p in (data_dir / sub).rglob("*") if (data_dir / sub).is_dir() else []:
            if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png") and sha256_file(p) in digests:
                n += 1
    return n


def guard(pred_files: list[str], calib_files: set[str], public_dir: Path, data_dir: Path, calib_rows) -> list[str]:
    errs = []
    files = [f.replace("\\", "/") for f in pred_files]
    outside = [f for f in files if f not in calib_files]
    if outside:
        errs.append(f"{len(outside)} ảnh của du_doan.csv không thuộc tập hiệu chỉnh (ví dụ {outside[0]}).")
    tests = test_files(public_dir) & set(files)
    if tests:
        errs.append(f"{len(tests)} ảnh thuộc TẬP ĐO (CocoGlide/test_korus), ví dụ {sorted(tests)[0]}.")
    dz = [f for f in files if f.split("/")[0] in ("that", "sua", "doi_chung")]
    if dz:
        errs.append(f"{len(dz)} ảnh có dạng đường dẫn của data.zip / A–D (that/, sua/, doi_chung/), ví dụ {dz[0]}.")
    n = datazip_overlap(calib_rows, data_dir)
    if n:
        errs.append(f"{n} ảnh của data.zip trùng byte với ảnh của tập hiệu chỉnh.")
    return errs


# ---------------------------------------------------------------------------------------------- proposal
def read_predictions(path: Path) -> tuple[list[dict], dict]:
    text = Path(path).read_text(encoding="utf-8-sig")
    reader = csv.DictReader(io.StringIO(text))
    if SCORE_COL not in (reader.fieldnames or []):
        raise dl.PrepError(f"{path} không có cột {SCORE_COL} (không phải du_doan.csv của cấu hình DefakerOne2).")
    rows, skipped = [], defaultdict(int)
    for r in reader:
        if r.get("status") != "ok":
            skipped[r.get("status") or "?"] += 1
            continue
        s = (r.get(SCORE_COL) or "").strip()
        y = dl.label01(r.get("label"))
        if not s or y is None:
            skipped["không có điểm"] += 1
            continue
        rows.append({"file": r["file"], "label": y, "loai": r.get("loai") or "(chung)",
                     "nhom": r.get("nhom") or r["file"], "score": float(s)})
    return rows, dict(skipped)


def _rate(scores, t):
    k = sum(1 for s in scores if s >= t)
    return {"n": len(scores), "so_anh_vuot": k, "ty_le": (k / len(scores)) if scores else None, "ci95": wilson(k, len(scores))}


def propose(rows: list[dict]) -> dict:
    neg = defaultdict(list)
    pos = defaultdict(list)
    for r in rows:
        (neg if r["label"] == 0 else pos)[r["loai"]].append(r["score"])
    if not neg:
        raise dl.PrepError("Không có ảnh thật nào chấm được trong du_doan.csv: không chọn được ngưỡng.")
    per = {}
    for src, s in sorted(neg.items()):
        per[src] = {"n": len(s), "high_rieng": threshold_for_fpr([0] * len(s), s, TARGET_HIGH),
                    "low_rieng": threshold_for_fpr([0] * len(s), s, TARGET_LOW)}
    high_raw = max(v["high_rieng"] for v in per.values())
    low_raw = max(v["low_rieng"] for v in per.values())
    high = ceil9(min(high_raw, 1.0))
    low = min(ceil9(min(low_raw, 1.0)), high)
    all_neg = [x for v in neg.values() for x in v]
    pooled_high = ceil9(min(threshold_for_fpr([0] * len(all_neg), all_neg, TARGET_HIGH), 1.0))
    pooled_low = min(ceil9(min(threshold_for_fpr([0] * len(all_neg), all_neg, TARGET_LOW), 1.0)), pooled_high)
    warnings = []
    for src, v in per.items():
        hi, lo = _rate(neg[src], high), _rate(neg[src], low)
        v.update(fpr_tai_high=hi, fpr_tai_low=lo, vung_khong_ket_luan=(lo["so_anh_vuot"] - hi["so_anh_vuot"]) / v["n"],
                 fpr_tai_0_5=_rate(neg[src], 0.5))
        if v["n"] < MIN_REAL_PER_SOURCE:
            warnings.append(f"Nguồn {src} chỉ có {v['n']} ảnh thật (< {MIN_REAL_PER_SOURCE}): FPR 5 % ước lượng kém chính xác.")
        if hi["ty_le"] is not None and hi["ty_le"] > TARGET_HIGH + 1e-12:
            warnings.append(f"Nguồn {src}: không đạt FPR ≤ 5 % ngay cả ở high = {high} (nhiều ảnh thật có p_fake ≈ 1).")
    pos_stats = {}
    for t, s in sorted(pos.items()):
        hi, lo = _rate(s, high), _rate(s, low)
        pos_stats[t] = {"n": len(s), "tpr_tai_high": hi, "tpr_tai_low": lo,
                        "vung_khong_ket_luan": (lo["so_anh_vuot"] - hi["so_anh_vuot"]) / len(s), "tpr_tai_0_5": _rate(s, 0.5)}
    if not pos:
        warnings.append("Không có ảnh sửa/giả (chạy --chi-anh-that): không có tỷ lệ phát hiện trên tập hiệu chỉnh.")
    y = [r["label"] for r in rows]
    sc = [r["score"] for r in rows]
    pooled = {"n_that": len(all_neg), "n_sua": sum(len(v) for v in pos.values()),
              "fpr_tai_high": _rate(all_neg, high), "fpr_tai_low": _rate(all_neg, low), "fpr_tai_0_5": _rate(all_neg, 0.5)}
    if pos:
        all_pos = [x for v in pos.values() for x in v]
        pooled.update(tpr_tai_high=_rate(all_pos, high), tpr_tai_low=_rate(all_pos, low), tpr_tai_0_5=_rate(all_pos, 0.5),
                      auroc=auroc(y, sc), auroc_ci95=bootstrap_ci(auroc, y, sc, [r["nhom"] for r in rows]))
    return {"low": low, "high": high, "low_chua_lam_tron": low_raw, "high_chua_lam_tron": high_raw,
            "nguon_quyet_dinh_high": max(per, key=lambda k: per[k]["high_rieng"]),
            "phuong_an_gop_chung_tham_khao": {"low": pooled_low, "high": pooled_high,
                                              "ghi_chu": "FPR tính trên mọi ảnh thật gộp chung; KHÔNG dùng (quy tắc của repo là từng nguồn)."},
            "anh_that_theo_nguon": per, "anh_sua_theo_loai": pos_stats, "gop": pooled, "canh_bao": warnings}


def render_md(p: dict) -> str:
    pc = dl.pc
    d = p["de_xuat"]
    L = [f"# Đề xuất ngưỡng DefakerOne2 — {p['tao_luc']}", "",
         f"- **high = {d['high']}** (“Nghi vấn”), **low = {d['low']}** (từ low tới dưới high: “Không kết luận”).",
         f"- Quy tắc: {RULE_VI}",
         f"- Nguồn ảnh thật quyết định high: {d['nguon_quyet_dinh_high']}.",
         f"- Tập hiệu chỉnh: `{p['tap_hieu_chinh']['file_nhan']}` (SHA-256 {p['tap_hieu_chinh']['sha256'][:12]}…); "
         f"điểm: `{p['du_doan']['file']}` (SHA-256 {p['du_doan']['sha256'][:12]}…).",
         f"- Đã ghi vào cấu hình: {'có — ' + ', '.join(p['da_ghi']['configs']) if p.get('da_ghi') else 'CHƯA (chạy lại với --ghi)'}", "",
         "## Ảnh thật theo nguồn (tỷ lệ báo nhầm, khoảng tin cậy Wilson 95 %)", "",
         "| Nguồn | n | high riêng | FPR tại high | FPR tại low | Vùng “Không kết luận” | FPR tại 0,5 (cũ) |", "|---|---|---|---|---|---|---|"]
    for src, v in d["anh_that_theo_nguon"].items():
        ci = v["fpr_tai_high"]["ci95"]
        L.append(f"| {src} | {v['n']} | {v['high_rieng']:.6g} | {pc(v['fpr_tai_high']['ty_le'])} "
                 f"[{pc(ci[0])}–{pc(ci[1])}] | {pc(v['fpr_tai_low']['ty_le'])} | {pc(v['vung_khong_ket_luan'])} | "
                 f"{pc(v['fpr_tai_0_5']['ty_le'])} |")
    if d["anh_sua_theo_loai"]:
        L += ["", "## Ảnh sửa/giả theo loại (tỷ lệ phát hiện trên tập hiệu chỉnh)", "",
              "| Loại | n | TPR tại high | TPR tại low | Vùng “Không kết luận” | TPR tại 0,5 (cũ) |", "|---|---|---|---|---|---|"]
        for t, v in d["anh_sua_theo_loai"].items():
            L.append(f"| {t} | {v['n']} | {pc(v['tpr_tai_high']['ty_le'])} | {pc(v['tpr_tai_low']['ty_le'])} | "
                     f"{pc(v['vung_khong_ket_luan'])} | {pc(v['tpr_tai_0_5']['ty_le'])} |")
    g = d["gop"]
    L += ["", f"Gộp: {g['n_that']} ảnh thật, {g['n_sua']} ảnh sửa · FPR tại high {pc(g['fpr_tai_high']['ty_le'])} · "
              f"AUROC {dl.f3(g.get('auroc'))}{dl.ci_text(g.get('auroc_ci95'))}", ""]
    if d["canh_bao"] or p.get("loi"):
        L += ["## Cảnh báo", ""] + [f"- {w}" for w in d["canh_bao"] + p.get("loi", [])] + [""]
    L += ["## Giới hạn", "",
          "- Ngưỡng chọn trên dữ liệu công khai (Korus, DeepFakeFace, OpenFake), không phải ảnh của công ty; tỷ lệ báo nhầm "
          "trên ảnh thật thực tế (Zalo, Facebook, ảnh chụp màn hình) có thể khác.",
          "- Số liệu tác giả DefakerOne2 chưa bình duyệt độc lập; server chạy fp16 + nén 8-bit.",
          "- Phải đo lại trên các bộ test (tools\\DO_LAI_DEFAKER2.bat) sau khi ghi ngưỡng; không chỉnh ngưỡng theo kết quả test.", ""]
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------------------------------------- writing
def next_experiment_id(log_path: Path) -> str:
    n = 0
    if log_path.is_file():
        for line in log_path.read_text(encoding="utf-8").splitlines():
            try:
                eid = json.loads(line).get("experiment_id", "")
            except ValueError:
                continue
            if eid.startswith("D2T") and eid[3:5].isdigit():
                n = max(n, int(eid[3:5]))
    return f"D2T{n + 1:02d}_defaker2_nguong_hieu_chinh"


def check_target(cfg_path: Path) -> dict:
    """The config may be written: not the locked config.json, and DefakerOne2 is its primary model."""
    p = Path(cfg_path).resolve()
    if p == LOCKED_CONFIG.resolve() or p.name == "config.json":
        raise dl.PrepError("Từ chối ghi config.json đã khóa.")
    if not p.is_file():
        raise dl.PrepError(f"Không thấy {p}")
    data = json.loads(p.read_bytes().decode("utf-8-sig"))
    det = (data.get("detectors") or {}).get("defaker2") or {}
    if det.get("type") != "defaker2_http" or det.get("role") != "primary":
        raise dl.PrepError(f"{p.name}: không có DefakerOne2 làm mô hình chính (role primary); không ghi.")
    return data


def write_config(cfg_path: Path, low: float, high: float, calibration: dict, log) -> None:
    p = Path(cfg_path).resolve()
    data = check_target(p)
    raw = p.read_bytes()
    det = data["detectors"]["defaker2"]
    det["thresholds"] = {
        "low": low, "high": high,
        "status": (f"Đã hiệu chỉnh ngày {calibration['date']} trên tập hiệu chỉnh riêng (phần dev của dữ liệu công khai, "
                   f"tools/chon_nguong_defaker2.py): high: FPR ≤ 5 %, low: FPR ≤ 20 % trên TỪNG nguồn ảnh thật; giữa hai "
                   f"ngưỡng là 'Không kết luận'. Xem {calibration['proposal_file']} và {calibration['experiment_id']} "
                   "trong EXPERIMENT_LOG.jsonl. Chưa kiểm chứng trên dữ liệu công ty."),
        "calibration": calibration}
    notice = (data.get("display") or {}).get("notice_vi") or ""
    if "0,5" in notice and p.name == dl.CFG_CHINH.name:          # the old notice states the 0.5 author threshold
        data["display"]["notice_vi"] = ("Bản DefakerOne2 làm MÔ HÌNH CHÍNH: kết luận chỉ theo DefakerOne2, ngưỡng đã hiệu "
                                        "chỉnh trên tập hiệu chỉnh riêng (xem thẻ DefakerOne2). Ba bộ phát hiện cũ đã TẮT. "
                                        "Cấu hình này khác bản đã đánh giá; ngưỡng chưa kiểm chứng trên dữ liệu công ty. Cần "
                                        "bật server addons\\defaker2\\CHAY_DEFAKER2.bat trước khi phân tích.")
    text = json.dumps(data, ensure_ascii=False, indent=1) + "\n"
    if b"\r\n" in raw:
        text = text.replace("\n", "\r\n")
    p.write_bytes(text.encode("utf-8"))
    try:
        load_config(p)
    except ForgeryError as exc:
        p.write_bytes(raw)                                       # never leave a config the app cannot load
        raise dl.PrepError(f"{p.name}: cấu hình sau khi ghi không hợp lệ ({exc}); đã trả lại bản cũ.") from exc
    log(f"Đã ghi ngưỡng vào {p}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="tools\\CHON_NGUONG_DEFAKER2.bat",
                                 description="Đề xuất ngưỡng DefakerOne2 trên tập hiệu chỉnh riêng (không bao giờ trên tập test).")
    ap.add_argument("--du-doan", type=Path, help="du_doan.csv đã có của TẬP HIỆU CHỈNH (bỏ qua bước chấm)")
    ap.add_argument("--chi-anh-that", action="store_true", help="chỉ chấm ảnh thật (đủ để chọn ngưỡng)")
    ap.add_argument("--thu", type=int, metavar="N", help="chạy thử N ảnh (không bao giờ dùng để ghi ngưỡng)")
    ap.add_argument("--ghi", action="store_true", help="ghi ngưỡng vào config DefakerOne2 và thêm dòng EXPERIMENT_LOG.jsonl")
    ap.add_argument("--config-ghi", type=Path, action="append", help="config được ghi (mặc định config_defaker2_mesorch.json)")
    ap.add_argument("--config-cham", type=Path, default=dl.CFG_CHINH, help="config dùng để chấm (mặc định config_defaker2_chinh.json)")
    ap.add_argument("--cong-khai", type=Path, default=dl.PUBLIC_DIR, help="thư mục của tao_nhan_cong_khai.py")
    ap.add_argument("--data", type=Path, default=dl.DATA_DIR, help="thư mục data.zip đã giải nén (chỉ để kiểm tra không trùng)")
    ap.add_argument("--bao-cao", type=Path, default=dl.REPORTS, help=argparse.SUPPRESS)
    ap.add_argument("--luu-dem", type=Path, default=dl.CACHE_DIR, help=argparse.SUPPRESS)
    ap.add_argument("--experiment-log", type=Path, default=EXPERIMENT_LOG, help=argparse.SUPPRESS)
    a = ap.parse_args(argv)
    if a.ghi and a.thu:
        ap.error("--thu chỉ để chạy thử; không dùng cùng --ghi")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = a.bao_cao / f"hieu_chinh_{stamp}"
    log = dl.Log(run_dir / "nhat_ky.log")
    try:
        log(f"CHON_NGUONG_DEFAKER2 · ứng dụng {__version__} · Python {sys.version.split()[0]}")
        try:
            label_csv, calib_rows, split_info = None, None, None
            if a.du_doan is None:
                dl.prepare_public("hieu_chinh", a.cong_khai, log)
            label_csv, calib_rows, split_info = load_calibration(a.cong_khai)
            calib_files = {r["file"].replace("\\", "/") for r in calib_rows}
            scoring = {}
            if a.du_doan is None:
                cfg_cham = a.config_cham.resolve()
                det = load_config(cfg_cham).data["detectors"].get("defaker2") or {}
                if not (det.get("enabled") and det.get("role") == "primary"):
                    raise dl.PrepError(f"{cfg_cham.name}: cần DefakerOne2 bật, role primary.")
                rows = calib_rows
                if a.chi_anh_that:
                    rows = [r for r in rows if dl.label01(r.get("label")) == 0]
                if a.thu:
                    rows, note = dl.choose_rows(rows, a.thu)
                    log(f"[CHẠY THỬ] {note} — đề xuất KHÔNG dùng được để ghi ngưỡng.")
                use = label_csv if len(rows) == len(calib_rows) else \
                    dl.write_label_subset(rows, run_dir / "chi_tiet" / "nhan_dung_lan_nay.csv")
                est = dl.estimate_minutes(a.cong_khai, rows, mesorch=False)
                if est is not None:
                    log(f"{len(rows)} ảnh; ước tính khoảng {est:.0f} phút nếu chưa có điểm lưu đệm (GTX 1650).")
                health = dl.check_server(det["url"], log)
                if not health:
                    return 3
                rc, rep_dir, ram, slow = dl.run_eval(a.cong_khai, use, cfg_cham, a.luu_dem,
                                                     dl.listening_pid(urllib.parse.urlsplit(det["url"]).port), log)
                if rep_dir is None or not (rep_dir / "du_doan.csv").is_file():
                    log(f"[LỖI] Kiểm định không tạo được du_doan.csv (mã thoát {rc}). Gửi {log.path} để được hỗ trợ.")
                    return 2
                (run_dir / "chi_tiet").mkdir(parents=True, exist_ok=True)
                for name in ("du_doan.csv", "metrics.json", "bao_cao_kiem_dinh.md", "bao_cao_kiem_dinh.html"):
                    shutil.copy2(rep_dir / name, run_dir / "chi_tiet" / name)
                du_doan = run_dir / "chi_tiet" / "du_doan.csv"
                scoring = {"cau_hinh_cham": str(cfg_cham.relative_to(ROOT)) if ROOT in cfg_cham.parents else str(cfg_cham),
                           "cau_hinh_cham_sha256": sha256_file(cfg_cham), "ma_thoat": rc, "ram": ram,
                           "server": {k: health.get(k) for k in ("device", "mode", "pixels_policy", "model") if k in health},
                           "anh_qua_60s": len(slow)}
            else:
                du_doan = a.du_doan.resolve()
                if not du_doan.is_file():
                    raise dl.PrepError(f"Không thấy {du_doan}")
            preds, skipped = read_predictions(du_doan)
            errs = guard([r["file"] for r in preds], calib_files, a.cong_khai, a.data, calib_rows)
            if errs:
                for e in errs:
                    log(f"[LỖI] {e}")
                log("DỪNG: ngưỡng chỉ được chọn trên tập hiệu chỉnh riêng (không dùng data.zip, A–D hay phần test).")
                return 2
            d = propose(preds)
        except (dl.PrepError, ForgeryError, OSError, ValueError) as exc:
            log(f"[LỖI] {exc}")
            return 2
        proposal_path = a.bao_cao / PROPOSAL_NAME
        p = {"tao_luc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "script": "tools/chon_nguong_defaker2.py",
             "phien_ban_app": __version__, "quy_tac": RULE_VI, "muc_tieu": {"high_fpr": TARGET_HIGH, "low_fpr": TARGET_LOW},
             "tap_hieu_chinh": {"file_nhan": str(label_csv), "sha256": sha256_file(label_csv),
                                "tach_tap_json": str(a.cong_khai / tnck.SPLIT_FILE),
                                "tach_tap_json_sha256": sha256_file(a.cong_khai / tnck.SPLIT_FILE),
                                "mo_ta": (split_info.get("tap") or {}).get("hieu_chinh"),
                                "quy_tac_tach": (split_info.get("quy_tac") or {}).get("hieu_chinh")},
             "du_doan": {"file": str(du_doan), "sha256": sha256_file(du_doan), "n_dung": len(preds), "bo_qua": skipped,
                         **scoring},
             "chay_thu": bool(a.thu), "de_xuat": d, "loi": [], "da_ghi": None}
        a.bao_cao.mkdir(parents=True, exist_ok=True)
        log("")
        log(f"ĐỀ XUẤT: high = {d['high']} (“Nghi vấn”), low = {d['low']} (“Không kết luận” từ low tới dưới high).")
        for src, v in d["anh_that_theo_nguon"].items():
            log(f"   ảnh thật {src}: n={v['n']}, báo nhầm tại high {dl.pc(v['fpr_tai_high']['ty_le'])}, "
                f"tại low {dl.pc(v['fpr_tai_low']['ty_le'])}, tại 0,5 (cũ) {dl.pc(v['fpr_tai_0_5']['ty_le'])}")
        for t, v in d["anh_sua_theo_loai"].items():
            log(f"   ảnh sửa {t}: n={v['n']}, phát hiện tại high {dl.pc(v['tpr_tai_high']['ty_le'])}, "
                f"tại 0,5 (cũ) {dl.pc(v['tpr_tai_0_5']['ty_le'])}")
        for w in d["canh_bao"]:
            log(f"[CẢNH BÁO] {w}")
        def save_proposal():
            proposal_path.write_text(json.dumps(p, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
            proposal_path.with_suffix(".md").write_text(render_md(p), encoding="utf-8")

        rel = lambda q: (str(Path(q).resolve().relative_to(ROOT)).replace("\\", "/")
                         if ROOT in Path(q).resolve().parents else str(q))
        small = [s for s, v in d["anh_that_theo_nguon"].items() if v["n"] < MIN_REAL_TO_WRITE]
        targets = [Path(c) for c in (a.config_ghi or DEFAULT_WRITE)]
        if a.ghi and small:
            p["loi"].append(f"Không ghi: nguồn {', '.join(small)} có dưới {MIN_REAL_TO_WRITE} ảnh thật.")
        elif a.ghi:
            try:
                for c in targets:                       # every target checked BEFORE anything is written
                    check_target(c)
            except (dl.PrepError, ValueError) as exc:
                p["loi"].append(f"Không ghi: {exc}")
        if a.ghi and not p["loi"]:
            eid = next_experiment_id(a.experiment_log)
            date = datetime.now().strftime("%Y-%m-%d")
            p["da_ghi"] = {"configs": [rel(c) for c in targets], "experiment_id": eid, "ngay": date}
            save_proposal()                             # final content: its SHA-256 goes into the configs and the log
            cal = {"date": date, "experiment_id": eid, "proposal_file": rel(proposal_path),
                   "proposal_sha256": sha256_file(proposal_path), "rule": RULE_VI,
                   "n_real_by_source": {k: v["n"] for k, v in d["anh_that_theo_nguon"].items()},
                   "calibration_labels_sha256": p["tap_hieu_chinh"]["sha256"]}
            try:
                for c in targets:
                    write_config(c, d["low"], d["high"], cal, log)
            except dl.PrepError as exc:
                log(f"[LỖI] {exc}")
                p["da_ghi"], p["loi"] = None, [f"Ghi cấu hình lỗi: {exc}"]
                save_proposal()
                return 2
            entry = {"experiment_id": eid, "time_utc": p["tao_luc"],
                     "muc_dich": "Chọn ngưỡng low/high cho DefakerOne2 (mô hình chính) trên tập hiệu chỉnh riêng",
                     "ly_do": ("Ngưỡng 0,5 của tác giả báo nhầm trên ảnh thật nhỏ/nén lại (ví dụ selfie 0,34 MP p_fake "
                               "0,97); cần điểm vận hành FPR ≈ 5 % như các detector khác của repo."),
                     "tap_hieu_chinh": {"mo_ta": p["tap_hieu_chinh"]["quy_tac_tach"],
                                        "n_anh_that_theo_nguon": cal["n_real_by_source"],
                                        "n_anh_sua": d["gop"]["n_sua"],
                                        "file_nhan_sha256": p["tap_hieu_chinh"]["sha256"]},
                     "khong_dung": "data.zip, A–D, CocoGlide, test_korus (script kiểm tra và dừng nếu có)",
                     "quy_tac": RULE_VI, "ket_qua": {"low": d["low"], "high": d["high"],
                                                      "nguon_quyet_dinh_high": d["nguon_quyet_dinh_high"],
                                                      "fpr_gop_tai_high": d["gop"]["fpr_tai_high"]["ty_le"]},
                     "file_bang_chung": {cal["proposal_file"]: cal["proposal_sha256"],
                                         rel(du_doan): p["du_doan"]["sha256"]},
                     "ghi_vao": [rel(c) for c in targets], "config_json_da_khoa": "không đổi",
                     "scoring_code_changed": False}
            with a.experiment_log.open("a", encoding="utf-8") as f:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")
            log(f"Đã thêm {eid} vào {a.experiment_log.name}. Đo lại các bộ test: tools\\DO_LAI_DEFAKER2.bat tat_ca")
            if any(Path(c).resolve().name == dl.CFG_CHINH.name for c in targets):
                log("[LƯU Ý] Đã ghi cả config_defaker2_chinh.json: test cũ "
                    "tests/test_defaker2.py::test_main_config_file_uses_only_defaker2_with_author_threshold (kiểm tra "
                    "ngưỡng 0,5) sẽ phải cập nhật trong cùng PR ngưỡng.")
        else:
            for e in p["loi"]:
                log(f"[LỖI] {e}")
            if not a.ghi:
                log("Chưa ghi vào cấu hình. Muốn ghi: chạy lại với --ghi (có thể kèm --du-doan để không chấm lại).")
            save_proposal()
        log(f"Đề xuất: {proposal_path}")
        log(f"Nhật ký: {log.path}")
        return 0 if not p["loi"] else 2
    finally:
        log.close()


if __name__ == "__main__":
    raise SystemExit(main())
