"""Kiểm định trên ảnh CÓ NHÃN của công ty — chạy hoàn toàn tại máy, không gửi ảnh đi đâu, không sửa cấu hình.

    python -m ifd.company_eval <THU_MUC_DU_LIEU> [--labels nhan.csv] [--config config.json] [--output outputs]

Cách đặt dữ liệu (chọn một):
  1) Theo thư mục (tên không phân biệt hoa/thường, có dấu hay không đều được):
       <THU_MUC>/that/...          ảnh thật, chưa chỉnh sửa            (nhãn 0)  — cũng nhận: real, goc, anh_that
       <THU_MUC>/sua/...           ảnh đã chỉnh sửa hoặc tạo giả      (nhãn 1)  — cũng nhận: gia, fake, anh_sua
     Thư mục con cấp 1 bên dưới (vd. sua/ghep_anh, sua/anh_ai, that/dien_thoai) được dùng làm "loại" để báo cáo riêng.
  2) File CSV UTF-8 có cột: file,label[,loai][,nhom]
       file  = đường dẫn tương đối so với <THU_MUC>;  label = 0/1 (hoặc that/sua);
       nhom  = (tùy chọn) mã nhóm nguồn: các ảnh cùng một cảnh/ảnh gốc có cùng mã → khoảng tin cậy tính theo nhóm.

Kết quả: <output>/kiem_dinh/<ten_thu_muc>_<ma>/bao_cao_<thời điểm>/ gồm bao_cao_kiem_dinh.html, .md, metrics.json,
du_doan.csv.  Điểm của từng ảnh được lưu đệm (cache_predictions.jsonl) để chạy lại không phải chấm lại; quyết định luôn
được tính lại từ điểm với ngưỡng HIỆN TẠI của config.json.  Công cụ chỉ báo cáo; không tự đổi ngưỡng.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import html
import io
import json
import logging
import os
import sys
import time
import unicodedata
import uuid
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import __version__
from .analysis import config_hash, decision_config_hash, load_evaluated_configs
from .config import DEFAULT_CONFIG, load_config
from .detectors.base import TASKS, decide
from .errors import ForgeryError

try:
    from ifd_core.metrics import auroc, average_precision, bootstrap_ci, confusion, threshold_for_fpr
except ImportError:  # pragma: no cover - the delivered package always contains ifd_core/
    raise SystemExit("Thiếu thư mục ifd_core (hàm tính chỉ số). Hãy dùng bản đóng gói đầy đủ.")

EXTS = (".jpg", ".jpeg", ".png")
LABEL_NAMES = {0: {"that", "real", "goc", "anh_that", "authentic", "pristine"},
               1: {"sua", "gia", "fake", "anh_sua", "anh_gia", "edited", "manipulated", "forged"}}
ORDER = {"no_clear_sign": 1, "needs_review": 2, "suspicious": 3}
TYPE_TASK = {"pscc_onnx": "local_manipulation", "mesorch_onnx": "local_manipulation",
             "commfor_onnx": "ai_generated", "face_clip_onnx": "face_deepfake"}
DEC_VI = {"suspicious": "Nghi vấn", "needs_review": "Cần kiểm tra thêm", "no_clear_sign": "Chưa thấy dấu hiệu rõ",
          "not_applicable": "Không áp dụng", "": ""}
log = logging.getLogger("ifd.company_eval")


def norm(name: str) -> str:
    s = unicodedata.normalize("NFD", name.strip().lower()).replace("đ", "d")
    s = "".join(c for c in s if unicodedata.category(c) != "Mn")
    return "_".join(s.replace("-", " ").split())


def label_of(value: str) -> int | None:
    v = norm(str(value))
    if v in ("0", "0.0"):
        return 0
    if v in ("1", "1.0"):
        return 1
    for lab, names in LABEL_NAMES.items():
        if v in names:
            return lab
    return None


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def collect_folder(root: Path) -> tuple[list[dict], list[str]]:
    items, notes = [], []
    label_dirs = [(d, label_of(d.name)) for d in sorted(root.iterdir()) if d.is_dir()]
    used = [(d, lab) for d, lab in label_dirs if lab is not None]
    if not used:
        raise ForgeryError("Không thấy thư mục nhãn. Cần thư mục con 'that' (ảnh thật) và/hoặc 'sua' (ảnh sửa/giả) "
                           "bên trong thư mục dữ liệu, hoặc dùng --labels nhan.csv.")
    for d, lab in label_dirs:
        if lab is None:
            notes.append(f"Bỏ qua thư mục không rõ nhãn: {d.name}")
    stray = [p.name for p in root.iterdir() if p.is_file() and p.suffix.lower() in EXTS]
    if stray:
        notes.append(f"Bỏ qua {len(stray)} ảnh nằm trực tiếp trong thư mục dữ liệu (không có nhãn).")
    for d, lab in used:
        for p in sorted(d.rglob("*")):
            if not p.is_file():
                continue
            if p.suffix.lower() not in EXTS:
                continue
            rel = p.relative_to(root)
            sub = rel.parts[1] if len(rel.parts) > 2 else "(chung)"
            items.append({"file": rel.as_posix(), "path": p, "label": lab, "loai": sub, "nhom": None})
    return items, notes


def collect_csv(root: Path, csv_path: Path) -> tuple[list[dict], list[str]]:
    items, notes = [], []
    try:
        text = csv_path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as exc:
        raise ForgeryError(f"Không đọc được file nhãn {csv_path}: {exc}") from exc
    reader = csv.DictReader(io.StringIO(text))
    cols = {norm(c): c for c in (reader.fieldnames or [])}
    if "file" not in cols or "label" not in cols:
        raise ForgeryError("File nhãn CSV cần ít nhất 2 cột: file,label (tùy chọn: loai,nhom).")
    for i, row in enumerate(reader, start=2):
        rel = (row.get(cols["file"]) or "").strip()
        lab = label_of(row.get(cols["label"]) or "")
        if not rel or lab is None:
            notes.append(f"Dòng {i}: thiếu file hoặc nhãn không hợp lệ → bỏ qua.")
            continue
        p = (root / rel).resolve()
        if not p.is_relative_to(root.resolve()):
            notes.append(f"Dòng {i}: đường dẫn ra ngoài thư mục dữ liệu → bỏ qua.")
            continue
        loai = (row.get(cols["loai"]) if "loai" in cols else None) or "(chung)"
        nhom = (row.get(cols["nhom"]) if "nhom" in cols else None) or None
        items.append({"file": Path(rel).as_posix(), "path": p, "label": lab, "loai": loai.strip(), "nhom": nhom})
    return items, notes


def score_settings_key(cfg) -> str:
    """Hash of everything that can change a SCORE (not a decision): model manifest + non-threshold settings."""
    dets = {k: {kk: vv for kk, vv in d.items() if kk != "thresholds"} for k, d in cfg.data["detectors"].items()
            if d.get("enabled")}
    manifest = cfg.path("models/manifest.json")
    mh = sha256_file(manifest) if manifest.is_file() else "no-manifest"
    return hashlib.sha256((json.dumps(dets, sort_keys=True) + mh).encode()).hexdigest()[:16]


def analyze_all(items, cfg, cache_path: Path, key: str, progress=print) -> list[dict]:
    cache = {}
    if cache_path.exists():
        for l in cache_path.read_text(encoding="utf-8").splitlines():
            try:
                r = json.loads(l)
            except ValueError:
                continue
            cache[(r["file"], r["sha256"], r["score_key"])] = r
    analyzer = None
    out = []
    n = len(items)
    with cache_path.open("a", encoding="utf-8") as f:
        for i, it in enumerate(items, start=1):
            rec = {"file": it["file"], "label": it["label"], "loai": it["loai"], "nhom": it["nhom"]}
            try:
                digest = sha256_file(it["path"])
            except OSError as exc:
                rec.update(status="error", error=f"Không đọc được file: {exc}", sha256=None, scores={})
                out.append(rec)
                continue
            hit = cache.get((it["file"], digest, key))
            if hit is None:
                if analyzer is None:
                    from .analysis import Analyzer  # noqa: PLC0415 - heavy import only when something must be scored
                    progress("Đang nạp mô hình...")
                    analyzer = Analyzer(cfg)
                t0 = time.perf_counter()
                hit = {"file": it["file"], "sha256": digest, "score_key": key, "app_version": __version__}
                try:
                    rep = analyzer.analyze(it["path"])
                    hit.update(status="ok", error=None, decoded_rgb_sha256=rep.metadata["decoded_rgb_sha256"],
                               scores={r.key: r.score for r in rep.results},
                               faces_found={r.key: r.details.get("faces_found") for r in rep.results
                                            if r.task == "face_deepfake"},
                               size_wh=rep.metadata["image"]["display_size_wh"])
                except ForgeryError as exc:
                    hit.update(status="error", error=str(exc), scores={})
                except Exception as exc:  # keep going; the traceback goes to the console log
                    log.exception("Lỗi không mong đợi với %s", it["file"])
                    hit.update(status="error", error=f"Lỗi không mong đợi: {type(exc).__name__}: {exc}", scores={})
                hit["seconds"] = round(time.perf_counter() - t0, 3)
                f.write(json.dumps(hit, ensure_ascii=False) + "\n")
                f.flush()
                progress(f"[{i}/{n}] {it['file']} — {'OK' if hit['status'] == 'ok' else 'LỖI'} ({hit['seconds']:.1f} s)")
            rec.update({k: hit.get(k) for k in ("status", "error", "sha256", "decoded_rgb_sha256", "scores",
                                                 "faces_found", "seconds")})
            out.append(rec)
    return out


def decisions(rec: dict, th: dict) -> tuple[dict, str]:
    per = {}
    for k, t in th.items():
        s = (rec.get("scores") or {}).get(k)
        per[k] = "not_applicable" if s is None else decide(float(s), t["low"], t["high"])
    applicable = [d for d in per.values() if d in ORDER]
    overall = max(applicable, key=lambda d: ORDER[d], default="no_clear_sign")
    return per, overall


def dedup(records: list[dict]) -> tuple[list[dict], dict]:
    """Same bytes or same decoded pixels count once; the same image with BOTH labels is excluded."""
    info = {"duplicates_counted_once": 0, "conflicting_labels_excluded": []}
    by_key = defaultdict(list)
    for r in records:
        if r["status"] == "ok":
            by_key[r.get("decoded_rgb_sha256") or r["sha256"]].append(r)
    keep = []
    for k, rs in by_key.items():
        labels = {r["label"] for r in rs}
        if len(labels) > 1:
            info["conflicting_labels_excluded"].extend(r["file"] for r in rs)
            continue
        keep.append(rs[0])
        info["duplicates_counted_once"] += len(rs) - 1
    return keep, info


def group_of(r: dict) -> str:
    return r.get("nhom") or r["file"]


def det_block(recs, det, t) -> dict:
    y = np.array([r["label"] for r in recs])
    raw = [(r.get("scores") or {}).get(det) for r in recs]
    s = np.array([-1.0 if v is None else float(v) for v in raw])     # None (no face) => never flagged
    out = {"n_pos": int(y.sum()), "n_neg": int((y == 0).sum()), "applicable_fraction": float(np.mean([v is not None for v in raw]))
           if raw else None}
    if out["n_pos"] and out["n_neg"]:
        out["auroc"] = auroc(y, s)
        out["ap"] = average_precision(y, s)
        out["auroc_ci95"] = bootstrap_ci(auroc, y, s, [group_of(r) for r in recs])
    out["at_high"] = confusion(y, s, t["high"])
    out["at_low"] = confusion(y, s, t["low"])
    band = (s >= t["low"]) & (s < t["high"])
    out["abstain_fraction"] = float(band.mean()) if len(s) else None
    by = {}
    for lab in (0, 1):
        for loai in sorted({r["loai"] for r in recs if r["label"] == lab}):
            m = np.array([r["label"] == lab and r["loai"] == loai for r in recs])
            if m.any():
                by[f"{'that' if lab == 0 else 'sua'}/{loai}"] = {"n": int(m.sum()), "ge_high": float((s[m] >= t["high"]).mean()),
                                                                  "ge_low": float((s[m] >= t["low"]).mean())}
    out["by_type"] = by
    return out


def product_block(recs, th) -> dict:
    y = np.array([r["label"] for r in recs])
    dec = np.array([ORDER[decisions(r, th)[1]] for r in recs])
    out = {}
    for name, flag in (("nghi_van_la_duong", dec >= 3), ("tu_can_kiem_tra_tro_len_la_duong", dec >= 2)):
        tp, fp = int((flag & (y == 1)).sum()), int((flag & (y == 0)).sum())
        fn, tn = int((~flag & (y == 1)).sum()), int((~flag & (y == 0)).sum())
        d = lambda a, b: (a / b) if b else None
        out[name] = {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": d(tp, tp + fp), "recall": d(tp, tp + fn),
                     "fpr": d(fp, fp + tn), "fnr": d(fn, fn + tp), "f1": d(2 * tp, 2 * tp + fp + fn)}
    by = {}
    for lab in (0, 1):
        for loai in ["(tất cả)"] + sorted({r["loai"] for r in recs if r["label"] == lab}):
            m = np.array([r["label"] == lab and (loai == "(tất cả)" or r["loai"] == loai) for r in recs])
            if not m.any():
                continue
            by[f"{'that' if lab == 0 else 'sua'}/{loai}"] = {
                "n": int(m.sum()), "nghi_van": float((dec[m] >= 3).mean()), "can_kiem_tra_tro_len": float((dec[m] >= 2).mean()),
                "chua_thay_dau_hieu": float((dec[m] == 1).mean())}
    out["by_type"] = by
    return out


def reference_thresholds(recs, th, seed_text="20261012") -> dict:
    """Informational only: thresholds for a target FPR chosen on half A of the REAL images (split by group),
    checked on half B.  Never written into config.json."""
    half = lambda r: int(hashlib.sha256((seed_text + group_of(r)).encode()).hexdigest(), 16) % 2
    out = {}
    for det in th:
        A = [r for r in recs if half(r) == 0]
        B = [r for r in recs if half(r) == 1]
        sa = lambda rs: np.array([-1.0 if (r.get("scores") or {}).get(det) is None else float(r["scores"][det]) for r in rs])
        ya, yb = np.array([r["label"] for r in A]), np.array([r["label"] for r in B])
        s_a, s_b = sa(A), sa(B)
        if (ya == 0).sum() < 50 or (yb == 0).sum() < 50:
            out[det] = {"note": "Cần ≥ 50 ảnh thật ở mỗi nửa để ước lượng ngưỡng tham khảo."}
            continue
        rows = []
        for target in (0.01, 0.05, 0.10):
            t = threshold_for_fpr(ya, s_a, target)
            row = {"target_fpr": target, "threshold_from_half_A": t,
                   "fpr_half_B": float((s_b[yb == 0] >= t).mean())}
            if (yb == 1).any():
                row["tpr_half_B"] = float((s_b[yb == 1] >= t).mean())
            rows.append(row)
        out[det] = {"n_real_A": int((ya == 0).sum()), "n_real_B": int((yb == 0).sum()), "rows": rows}
    return out


def f3(x):
    return "—" if x is None else f"{x:.3f}".replace(".", ",")


def pc(x):
    return "—" if x is None else f"{100 * x:.1f} %".replace(".", ",")


def render(m: dict) -> tuple[str, str]:
    """Markdown + standalone HTML (no external resources)."""
    L = [f"# Báo cáo kiểm định trên dữ liệu có nhãn — {m['dataset_name']}", "",
         f"Thời điểm: {m['created_utc']} · phiên bản ứng dụng {m['app_version']} · cấu hình `{m['config_file']}` (SHA-256 {m['config_sha256'][:12]}…)", "",
         ("> **Cấu hình bộ phát hiện/ngưỡng KHÁC bản đã được đánh giá trong báo cáo chất lượng.**\n"
          if m.get("config_matches_evaluated") is False else ""),
         "> Kết quả này chỉ đúng cho **chính tập ảnh này** và cách gán nhãn của nó. Điểm không phải xác suất. "
         "Không chỉnh ngưỡng rồi đánh giá lại trên cùng tập ảnh; muốn đổi ngưỡng phải kiểm tra lại trên tập ảnh mới.", "",
         "## Dữ liệu", "",
         f"- Ảnh tìm thấy: {m['counts']['found']} · chấm được: {m['counts']['ok']} · lỗi đọc: {m['counts']['errors']}",
         f"- Ảnh thật: {m['counts']['real']} · ảnh sửa/giả: {m['counts']['fake']} (sau khi gộp trùng)",
         f"- Ảnh trùng (cùng bytes hoặc cùng pixel) tính một lần: {m['dedup']['duplicates_counted_once']} · "
         f"trùng nhưng khác nhãn (đã loại): {len(m['dedup']['conflicting_labels_excluded'])}"]
    L += [f"- Lưu ý: {w}" for w in m["warnings"]]
    L += ["", "## Kết luận người dùng nhìn thấy (cấp ứng dụng)", "",
          "| Nhóm | n | “Nghi vấn” | “Cần kiểm tra thêm” trở lên | “Chưa thấy dấu hiệu rõ” |", "|---|---|---|---|---|"]
    for k, b in m["product"]["by_type"].items():
        L.append(f"| {k} | {b['n']} | {pc(b['nghi_van'])} | {pc(b['can_kiem_tra_tro_len'])} | {pc(b['chua_thay_dau_hieu'])} |")
    L += ["", "Đếm dương tính = ảnh bị gắn mức đó (ảnh sửa/giả là lớp dương):", "",
          "| Cách đếm | TP | FP | TN | FN | Precision | Recall (TPR) | FPR | FNR |", "|---|---|---|---|---|---|---|---|---|"]
    for name, title in (("nghi_van_la_duong", "“Nghi vấn”"), ("tu_can_kiem_tra_tro_len_la_duong", "“Cần kiểm tra thêm” trở lên")):
        b = m["product"][name]
        L.append(f"| {title} | {b['tp']} | {b['fp']} | {b['tn']} | {b['fn']} | {pc(b['precision'])} | {pc(b['recall'])} | {pc(b['fpr'])} | {pc(b['fnr'])} |")
    L += ["", "## Từng bộ phát hiện", "",
          "Mỗi bộ phát hiện chỉ nhắm một loại; ví dụ bộ “ảnh sinh hoàn toàn bằng AI” không được kỳ vọng bắt ảnh ghép. "
          "Hãy đọc dòng theo đúng “loại” ảnh sửa ở bảng nhóm bên dưới mỗi bộ.", ""]
    for det, b in m["detectors"].items():
        t = m["thresholds"][det]
        task = m.get("detector_tasks", {}).get(det, "")
        L += [f"### {det} — {TASKS.get(task, task)} (low {f3(t['low'])} / high {f3(t['high'])})", ""]
        ci = b.get("auroc_ci95")
        L.append(f"- AUROC: {f3(b.get('auroc'))}" + (f" [CI 95 %: {f3(ci[0])}–{f3(ci[1])}]" if ci else "") +
                 f" · AP: {f3(b.get('ap'))} · tỷ lệ áp dụng được: {pc(b.get('applicable_fraction'))} · vùng “cần kiểm tra”: {pc(b.get('abstain_fraction'))}")
        L += ["", "| Ngưỡng | TP | FP | TN | FN | Precision | TPR | FPR | FNR |", "|---|---|---|---|---|---|---|---|---|"]
        for key, title in (("at_high", "high (“Nghi vấn”)"), ("at_low", "low (“Cần kiểm tra thêm”)")):
            c = b[key]
            L.append(f"| {title} | {c['tp']} | {c['fp']} | {c['tn']} | {c['fn']} | {pc(c['precision'])} | {pc(c['recall'])} | {pc(c['fpr'])} | {pc(c['fnr'])} |")
        L += ["", "| Nhóm | n | ≥ high | ≥ low |", "|---|---|---|---|"]
        for k, v in b["by_type"].items():
            L.append(f"| {k} | {v['n']} | {pc(v['ge_high'])} | {pc(v['ge_low'])} |")
        L.append("")
    L += ["## Ngưỡng tham khảo theo ảnh thật của chính tập này (KHÔNG tự áp dụng)", "",
          "Chọn trên nửa A (chia ngẫu nhiên theo nhóm), kiểm tra trên nửa B. Chỉ để thảo luận; muốn áp dụng phải ghi quyết định và kiểm tra lại trên tập mới.", ""]
    for det, r in m["reference_thresholds"].items():
        if "rows" not in r:
            L.append(f"- {det}: {r['note']}")
            continue
        L += [f"**{det}** (ảnh thật: nửa A {r['n_real_A']}, nửa B {r['n_real_B']})", "",
              "| FPR mục tiêu (nửa A) | Ngưỡng | FPR thực tế nửa B | TPR nửa B |", "|---|---|---|---|"]
        for row in r["rows"]:
            L.append(f"| {pc(row['target_fpr'])} | {row['threshold_from_half_A']:.6g} | {pc(row['fpr_half_B'])} | {pc(row.get('tpr_half_B'))} |")
        L.append("")
    if m["errors"]:
        L += ["", "## Ảnh lỗi", ""] + [f"- `{e['file']}`: {e['error']}" for e in m["errors"][:200]]
    md = "\n".join(L) + "\n"
    body = []
    in_table = False
    for line in L:
        if line.startswith("|"):
            cells = [html.escape(c.strip()) for c in line.strip("|").split("|")]
            if set("".join(cells)) <= set("-: "):
                continue
            if not in_table:
                body.append("<table>")
                in_table = True
                body.append("<tr>" + "".join(f"<th>{c}</th>" for c in cells) + "</tr>")
            else:
                body.append("<tr>" + "".join(f"<td>{c}</td>" for c in cells) + "</tr>")
            continue
        if in_table:
            body.append("</table>")
            in_table = False
        if line.startswith("### "):
            body.append(f"<h3>{html.escape(line[4:])}</h3>")
        elif line.startswith("## "):
            body.append(f"<h2>{html.escape(line[3:])}</h2>")
        elif line.startswith("# "):
            body.append(f"<h1>{html.escape(line[2:])}</h1>")
        elif line.startswith("> "):
            body.append(f"<p class='note'>{html.escape(line[2:])}</p>")
        elif line.startswith("- "):
            body.append(f"<p>• {html.escape(line[2:])}</p>")
        elif line.strip():
            body.append(f"<p>{html.escape(line)}</p>")
    if in_table:
        body.append("</table>")
    page = ("<!doctype html><html lang='vi'><head><meta charset='utf-8'><title>Báo cáo kiểm định</title><style>"
            "body{font-family:Segoe UI,Arial,sans-serif;max-width:1100px;margin:24px auto;padding:0 16px;color:#1d2433}"
            "table{border-collapse:collapse;margin:8px 0 16px}td,th{border:1px solid #c9ced8;padding:4px 8px;font-size:14px}"
            "th{background:#eef1f6}.note{background:#fff6e0;border-left:4px solid #e0a000;padding:8px}</style></head><body>"
            + "\n".join(body) + "</body></html>\n")
    return md, page


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m ifd.company_eval",
                                 description="Kiểm định ứng dụng trên ảnh có nhãn của công ty (chạy tại máy).")
    ap.add_argument("data", type=Path, help="Thư mục dữ liệu (có that/ và sua/, hoặc dùng --labels)")
    ap.add_argument("--labels", type=Path, default=None, help="CSV: file,label[,loai][,nhom]")
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    ap.add_argument("--output", type=Path, default=Path("outputs"))
    ap.add_argument("--open", action="store_true", help="Mở báo cáo HTML bằng trình duyệt mặc định khi xong (Windows)")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    try:
        cfg = load_config(args.config)
        root = args.data.resolve()
        if not root.is_dir():
            raise ForgeryError(f"Không tìm thấy thư mục dữ liệu: {args.data}")
        output = args.output.resolve()
        if output == root or root in output.parents:
            raise ForgeryError("Thư mục kết quả không được nằm trong thư mục dữ liệu.")
        items, notes = collect_csv(root, args.labels) if args.labels else collect_folder(root)
        items = [it for it in items if not Path(it["path"]).resolve().is_relative_to(output)]
        if not items:
            raise ForgeryError("Không có ảnh .jpg/.jpeg/.png nào có nhãn để kiểm định.")
        th = {k: d["thresholds"] for k, d in cfg.data["detectors"].items() if d.get("enabled")}
        key = score_settings_key(cfg)
        work = output / "kiem_dinh" / f"{norm(root.name)[:40] or 'du_lieu'}_{hashlib.sha256(str(root).encode()).hexdigest()[:8]}"
        work.mkdir(parents=True, exist_ok=True)
        print(f"Kiểm định {len(items)} ảnh có nhãn; ảnh KHÔNG rời khỏi máy. Điểm lưu đệm tại {work}", flush=True)
        recs = analyze_all(items, cfg, work / "cache_predictions.jsonl", key, progress=lambda s: print(s, flush=True))
        errors = [{"file": r["file"], "error": r["error"]} for r in recs if r["status"] != "ok"]
        kept, dd = dedup(recs)
        warnings = list(notes)
        n_real, n_fake = sum(r["label"] == 0 for r in kept), sum(r["label"] == 1 for r in kept)
        if n_real < 200 or n_fake < 200:
            warnings.append("Ít ảnh (khuyến nghị ≥ 200 ảnh thật và ≥ 200 ảnh sửa/giả cho mỗi loại quan tâm): "
                            "khoảng tin cậy rộng, chưa đủ để kết luận cho vận hành.")
        if not any(r.get("nhom") for r in kept):
            warnings.append("Không có cột 'nhom': khoảng tin cậy coi mỗi ảnh độc lập; nếu nhiều ảnh cùng một cảnh/ảnh gốc, "
                            "CI thực tế sẽ rộng hơn.")
        m = {"dataset_name": root.name, "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
             "app_version": __version__, "config_file": str(Path(args.config)), "config_sha256": config_hash(cfg),
             "score_key": key, "thresholds": th,
             "config_matches_evaluated": (None if not load_evaluated_configs(cfg.base_dir) else
                                          decision_config_hash(cfg) in load_evaluated_configs(cfg.base_dir)),
             "evaluated_profile": load_evaluated_configs(cfg.base_dir).get(decision_config_hash(cfg)),
             "detector_tasks": {k: TYPE_TASK.get(d["type"], "") for k, d in cfg.data["detectors"].items() if d.get("enabled")},
             "counts": {"found": len(items), "ok": sum(r["status"] == "ok" for r in recs), "errors": len(errors),
                        "real": n_real, "fake": n_fake},
             "dedup": dd, "warnings": warnings, "errors": errors,
             "detectors": {k: det_block(kept, k, t) for k, t in th.items()} if kept else {},
             "product": product_block(kept, th) if kept else {"by_type": {}},
             "reference_thresholds": reference_thresholds(kept, th) if kept else {}}
        if not kept:
            raise ForgeryError("Không có ảnh nào chấm được (xem danh sách lỗi phía trên).")
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        rep_dir = work / f"bao_cao_{stamp}_{uuid.uuid4().hex[:6]}"       # a new folder per report, never overwritten
        rep_dir.mkdir(parents=True, exist_ok=False)
        md, page = render(m)
        (rep_dir / "metrics.json").write_text(json.dumps(m, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
        (rep_dir / "bao_cao_kiem_dinh.md").write_text(md, encoding="utf-8")
        (rep_dir / "bao_cao_kiem_dinh.html").write_text(page, encoding="utf-8")
        buf = io.StringIO()
        cols = ["file", "label", "loai", "nhom", "status", "overall_decision", "overall_decision_vi"]
        for k in th:
            cols += [f"{k}_score", f"{k}_decision"]
        cols += ["error"]
        w = csv.DictWriter(buf, fieldnames=cols)
        w.writeheader()
        kept_ids = {id(r) for r in kept}
        for r in recs:
            row = {c: "" for c in cols}
            row.update(file=r["file"], label=r["label"], loai=r["loai"], nhom=r.get("nhom") or "", status=r["status"],
                       error=r.get("error") or "")
            if r["status"] == "ok":
                per, overall = decisions(r, th)
                if id(r) not in kept_ids:
                    row["status"] = "duplicate_or_conflict"
                row.update(overall_decision=overall, overall_decision_vi=DEC_VI[overall])
                for k in th:
                    s = (r.get("scores") or {}).get(k)
                    row[f"{k}_score"] = "" if s is None else s
                    row[f"{k}_decision"] = per[k]
            w.writerow(row)
        (rep_dir / "du_doan.csv").write_bytes(("﻿" + buf.getvalue()).encode("utf-8"))
        print(f"Xong. Báo cáo: {rep_dir / 'bao_cao_kiem_dinh.html'}", flush=True)
        if args.open and os.name == "nt":
            try:
                os.startfile(str(rep_dir / "bao_cao_kiem_dinh.html"))  # local file, default browser; nothing is sent anywhere
            except OSError:
                pass
        return 1 if errors else 0
    except ForgeryError as exc:
        print(f"Lỗi: {exc}", file=sys.stderr)
        return 2
    except OSError as exc:
        print(f"Lỗi hệ thống tệp: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
