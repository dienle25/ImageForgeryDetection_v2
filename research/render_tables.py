"""Render metrics JSON (research/final_report.py) into Markdown tables — numbers are never typed by hand.

  python research/render_tables.py --v2 metrics_test_v2.json [--strict metrics_test_v2_strict.json]
      [--v1 metrics_test_v1.json] [--robust robust_summary.json] --out docs/_tables_test.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def f(x, d=3):
    if x is None:
        return "—"
    if isinstance(x, (int,)) and not isinstance(x, bool):
        return str(x)
    return f"{x:.{d}f}".replace(".", ",")


def pct(x, d=1):
    return "—" if x is None else f"{100 * x:.{d}f} %".replace(".", ",")


def ci(b):
    c = b.get("auroc_ci95")
    return f"{f(b.get('auroc'))} [{f(c[0])}; {f(c[1])}]" if c else f(b.get("auroc"))


def conf_cells(b, key):
    c = b.get(key, {})
    return [pct(c.get("recall")), pct(c.get("fpr")), pct(c.get("precision")), f(c.get("f1"))]


def A_tables(m):
    A = m["tasks"].get("local_manipulation", {})
    out = ["#### Mức ảnh", "",
           "| Tập test | Sửa / Gốc | AUROC [CI 95 %] | AP | TPR @high | FPR @high | Precision @high | F1 @high | TPR @low | FPR @low | Vùng “cần kiểm tra” | Coverage |",
           "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    names = {"ctrl_mei": "ctrl_mei test (script, camera thật)", "korus": "Korus test (sửa tay thực tế)", "cocoglide": "CocoGlide (inpainting GLIDE, ngoài hoàn toàn)"}
    for k in ("ctrl_mei", "korus", "cocoglide"):
        if k not in A:
            continue
        b = A[k]
        hi = conf_cells(b, "at_high")
        lo = b.get("at_low", {})
        out.append(f"| {names[k]} | {b['n_pos']} / {b['n_neg']} | {ci(b)} | {f(b.get('ap'))} | {hi[0]} | {hi[1]} | {hi[2]} | {hi[3]} | "
                   f"{pct(lo.get('recall'))} | {pct(lo.get('fpr'))} | {pct(b.get('abstain_fraction'))} | {pct(b.get('coverage'))} |")
    out += ["", "#### Mức pixel (ngưỡng pixel cố định 0,5; ảnh gốc báo riêng — chính sách mask rỗng ở DATA_PROTOCOL §8)", "",
            "| Tập test | Ảnh có mask | F1 macro | IoU macro | F1 micro | AUC pixel TB | % ảnh F1 ≥ 0,5 | Kích hoạt nhầm TB trên ảnh gốc | % ảnh gốc kích hoạt > 1 % |",
            "|---|---|---|---|---|---|---|---|---|"]
    for k in ("ctrl_mei", "korus", "cocoglide"):
        if k not in A:
            continue
        p = A[k]["pixel"]
        out.append(f"| {names[k]} | {p.get('n_masked_positives')} | {f(p.get('macro_f1'))} | {f(p.get('macro_iou'))} | {f(p.get('micro_f1'))} | "
                   f"{f(p.get('mean_pixel_auc'))} | {pct(p.get('fraction_f1_ge_0_5'))} | {pct(p.get('authentic_mean_false_activation'), 2)} | "
                   f"{pct(p.get('authentic_fraction_images_activation_gt_1pct'))} |")
    if "ctrl_mei" in A:
        for field, title in (("by_operation", "thao tác"), ("by_size_bucket", "kích thước vùng sửa"), ("by_post_processing", "hậu xử lý (cặp gốc cùng hậu xử lý)")):
            sub = A["ctrl_mei"].get(field)
            if not sub:
                continue
            out += ["", f"ctrl_mei test theo {title}:", "", "| Nhóm | n sửa | AUROC | TPR @high | FPR @high |", "|---|---|---|---|---|"]
            for k, b in sub.items():
                out.append(f"| {k} | {b['n_pos']} | {ci(b)} | {pct(b.get('at_high', {}).get('recall'))} | {pct(b.get('at_high', {}).get('fpr'))} |")
    if "korus" in A and A["korus"].get("by_camera"):
        out += ["", "Korus test theo máy ảnh:", "", "| Máy | n sửa / gốc | AUROC | TPR @high | FPR @high |", "|---|---|---|---|---|"]
        for k, b in A["korus"]["by_camera"].items():
            out.append(f"| {k} | {b['n_pos']} / {b['n_neg']} | {f(b.get('auroc'))} | {pct(b.get('at_high', {}).get('recall'))} | {pct(b.get('at_high', {}).get('fpr'))} |")
    return out


def B_tables(m):
    B = m["tasks"].get("ai_generated", {})
    out = []
    of = B.get("openfake_holdout_generators")
    if of:
        hi = conf_cells(of, "at_high")
        out += ["| Tập | Sinh / Thật | AUROC [CI 95 %] | AP | TPR @high | FPR @high | Precision @high | TPR @low | FPR @low | Vùng “cần kiểm tra” |",
                "|---|---|---|---|---|---|---|---|---|---|",
                f"| OpenFake test — {len(of['by_generator'])} generator không có trong dev | {of['n_pos']} / {of['n_neg']} | {ci(of)} | {f(of.get('ap'))} | {hi[0]} | {hi[1]} | {hi[2]} | "
                f"{pct(of['at_low'].get('recall'))} | {pct(of['at_low'].get('fpr'))} | {pct(of.get('abstain_fraction'))} |"]
        d = B.get("dff_text2img_vs_wiki")
        if d:
            hd = conf_cells(d, "at_high")
            out.append(f"| DeepFakeFace text2img (SD 1.5) vs ảnh thật | {d['n_pos']} / {d['n_neg']} | {ci(d)} | {f(d.get('ap'))} | {hd[0]} | {hd[1]} | {hd[2]} | "
                       f"{pct(d['at_low'].get('recall'))} | {pct(d['at_low'].get('fpr'))} | {pct(d.get('abstain_fraction'))} |")
        out += ["", "Theo generator (so với toàn bộ ảnh thật OpenFake test):", "", "| Generator | n | AUROC | TPR @high | TPR @low |", "|---|---|---|---|---|"]
        for g, b in sorted(of["by_generator"].items(), key=lambda kv: -(kv[1].get("auroc") or 0)):
            out.append(f"| {g} | {b['n_pos']} | {ci(b)} | {pct(b['at_high'].get('recall'))} | {pct(b['at_low'].get('recall'))} |")
    cam = B.get("false_positive_on_real_photos_by_source", {})
    if cam:
        out += ["", "Báo nhầm “ảnh sinh AI” trên ảnh thật theo nguồn:", "", "| Nguồn ảnh thật | n | FPR @high | ≥ low (cần kiểm tra trở lên) |", "|---|---|---|---|"]
        for k, b in cam.items():
            out.append(f"| {k} | {b['n']} | {pct(b['fpr_at_high'])} | {pct(b['flag_or_review_at_low'])} |")
    cross = B.get("activation_on_partially_edited_images_(not_errors)", {})
    if cross:
        out += ["", "Kích hoạt trên ảnh chỉ sửa một phần (không tính đúng/sai cho bài toán B):", "", "| Nguồn | n | % ≥ high |", "|---|---|---|"]
        for k, b in cross.items():
            out.append(f"| {k} | {b['n']} | {pct(b['fraction_ge_high'])} |")
    return out


def C_tables(m):
    C = m["tasks"].get("face_deepfake", {})
    out = ["| So sánh | Giả / Thật | AUROC [CI 95 %] | AP | TPR @high | FPR @high | Precision @high | TPR @low | FPR @low | Vùng “cần kiểm tra” |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    names = {"faceswap_insightface_vs_real": "Hoán đổi mặt (InsightFace) vs mặt thật",
             "face_inpainting_sd_vs_real": "Inpainting mặt (SD) vs mặt thật",
             "fully_generated_faces_vs_real_(cross_task)": "Mặt sinh hoàn toàn (SD 1.5) vs mặt thật — chéo bài toán"}
    for k, nm in names.items():
        if k in C:
            b = C[k]
            hi = conf_cells(b, "at_high")
            out.append(f"| {nm} | {b['n_pos']} / {b['n_neg']} | {ci(b)} | {f(b.get('ap'))} | {hi[0]} | {hi[1]} | {hi[2]} | "
                       f"{pct(b['at_low'].get('recall'))} | {pct(b['at_low'].get('fpr'))} | {pct(b.get('abstain_fraction'))} |")
    if "no_face_found_rate_on_real_faces" in C:
        out += ["", f"Tỷ lệ không phát hiện được mặt trên ảnh chân dung thật: {pct(C['no_face_found_rate_on_real_faces'])}."]
    nf = C.get("false_positive_on_other_real_images", {})
    if nf:
        out += ["", "| Ảnh thật khác | n | % có mặt được phát hiện | FPR @high |", "|---|---|---|---|"]
        for k, b in nf.items():
            out.append(f"| {k} | {b['n']} | {pct(b['faces_found_fraction'])} | {pct(b['fpr_at_high'])} |")
    return out


def product_table(m, title):
    P = m.get("product_level", {})
    out = [f"| Nguồn ({title}) | n thật / giả | Thật → “Nghi vấn” | Thật → “Cần kiểm tra” trở lên | Giả → “Nghi vấn” | Giả → “Cần kiểm tra” trở lên | Giả → bỏ sót | Bộ phát hiện gây “Nghi vấn” | Thời gian p50 / p95 |",
           "|---|---|---|---|---|---|---|---|---|"]
    for k, b in P.items():
        sec = b.get("seconds_p50_p95")
        out.append(f"| {k} | {b.get('n_real', 0)} / {b.get('n_manipulated_or_generated', 0)} | {pct(b.get('real_flagged_suspicious'))} | "
                   f"{pct(b.get('real_needs_review_or_more'))} | {pct(b.get('fake_flagged_suspicious'))} | {pct(b.get('fake_needs_review_or_more'))} | "
                   f"{pct(b.get('fake_missed_no_clear_sign'))} | {', '.join(f'{kk}: {vv}' for kk, vv in b.get('suspicious_triggered_by', {}).items()) or '—'} | "
                   f"{(f(sec[0], 2) + ' s / ' + f(sec[1], 2) + ' s') if sec else '—'} |")
    return out


def robust_tables(r):
    out = ["| Bộ phát hiện | Biến đổi | n thật / giả | AUROC gốc → sau | Thật ≥ high: gốc → sau | Thật ≥ low: gốc → sau | "
           "Giả ≥ high: gốc → sau | Thật vượt lên “Nghi vấn” | Ghi chú |",
           "|---|---|---|---|---|---|---|---|---|"]
    for det, d in r["results"].items():
        for tr, b in d.items():
            neg = b.get("real_or_authentic", {})
            pos = b.get("manipulated_or_generated", {})
            au = f"{f(b.get('auroc_original'))} → {f(b.get('auroc_transformed'))}" if "auroc_original" in b else "—"
            note = (f"không thấy mặt: {pct(neg.get('no_face_original'))} → {pct(neg.get('no_face_transformed'))}"
                    if "no_face_original" in neg else "")
            out.append(f"| {det} | {tr} | {b['n_neg']} / {b['n_pos']} | {au} | "
                       f"{pct(neg.get('ge_high_original'))} → {pct(neg.get('ge_high_transformed'))} | "
                       f"{pct(neg.get('ge_low_original'))} → {pct(neg.get('ge_low_transformed'))} | "
                       f"{pct(pos.get('ge_high_original'))} → {pct(pos.get('ge_high_transformed'))} | "
                       f"{neg.get('crossed_up_to_suspicious', '—')} | {note} |")
    return out


def ba_tables(b):
    out = [f"Ảnh chấm được bởi CẢ HAI phiên bản: {b['n_paired_images']}", "",
           "#### Bài toán A — cùng ảnh, AUROC mức ảnh (chênh lệch: bootstrap ghép cặp theo nhóm nguồn)", "",
           "| Tập | Sửa / Gốc | v1 PSCC-Net | v2 Mesorch | Chênh lệch v2 − v1 [CI 95 %] | F1 pixel macro v1 → v2 | Kích hoạt nhầm TB trên ảnh gốc v1 → v2 |",
           "|---|---|---|---|---|---|---|"]
    names = {"ctrl_mei": "ctrl_mei test", "korus": "Korus test", "cocoglide": "CocoGlide"}
    for k, x in b["task_a"].items():
        ci_ = x.get("difference_ci95")
        pf, fa = x.get("pixel_macro_f1", {}), x.get("false_activation_on_authentic", {})
        out.append(f"| {names.get(k, k)} | {x['n_pos']} / {x['n_neg']} | {f(x['auroc_v1_pscc'])} | {f(x['auroc_v2_mesorch'])} | "
                   f"{f(x['difference_v2_minus_v1'])} [{f(ci_[0]) if ci_ else '—'}; {f(ci_[1]) if ci_ else '—'}] | "
                   f"{f(pf.get('v1'))} → {f(pf.get('v2'))} | {pct(fa.get('v1_mean_fraction'))} → {pct(fa.get('v2_mean_fraction'))} |")
    out += ["", "#### Kết luận người dùng nhìn thấy — cùng ảnh (khoảng Wilson 95 %)", "",
            "| Nguồn | Loại ảnh | n | “Nghi vấn” v1 | “Nghi vấn” v2 | “Cần kiểm tra” trở lên v1 | “Cần kiểm tra” trở lên v2 | Chỉ v1 gắn / chỉ v2 gắn “Nghi vấn” |",
            "|---|---|---|---|---|---|---|---|"]

    def pci(v, c):
        return f"{pct(v)} [{pct(c[0])}–{pct(c[1])}]" if c else pct(v)
    for src, blk in b["by_source"].items():
        for name, title in (("real", "thật"), ("manipulated_or_generated", "sửa/sinh/giả")):
            if name not in blk:
                continue
            x = blk[name]
            ps = x["paired_suspicious"]
            out.append(f"| {src} | {title} | {x['n']} | {pci(x['v1']['suspicious'], x['v1']['suspicious_ci95'])} | "
                       f"{pci(x['v2']['suspicious'], x['v2']['suspicious_ci95'])} | {pct(x['v1']['review_or_more'])} | "
                       f"{pct(x['v2']['review_or_more'])} | {ps['v1_only']} / {ps['v2_only']} |")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v2", type=Path, required=True)
    ap.add_argument("--strict", type=Path, default=None)
    ap.add_argument("--v1", type=Path, default=None)
    ap.add_argument("--robust", type=Path, default=None)
    ap.add_argument("--ba", type=Path, default=None, help="research/before_after_compare.py output")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args()
    m = json.loads(a.v2.read_text(encoding="utf-8"))
    lines = [f"<!-- generated by research/render_tables.py from {a.v2.name}; do not edit numbers by hand -->", "",
             f"Ảnh được chấm: {m['n_images']} · lỗi đọc: {len(m['errors'])} · trùng pixel tính một lần: {m['identical_pixel_duplicates_skipped']} · "
             f"loại trừ theo manifest: {m['excluded_by_manifest']}", "",
             "Ngưỡng dùng: " + "; ".join(f"`{k}` low {f(v['low'], 4)} / high {f(v['high'], 4)}" for k, v in m["thresholds"].items()), "",
             "### A. Chỉnh sửa cục bộ (Mesorch)", ""] + A_tables(m) + ["", "### B. Ảnh sinh hoàn toàn bằng AI (Community Forensics)", ""] + \
        B_tables(m) + ["", "### C. Deepfake khuôn mặt (YuNet + CLIP)", ""] + C_tables(m) + \
        ["", "### Cấp ứng dụng — kết luận người dùng nhìn thấy (mức cao nhất trong các bài toán)", ""] + product_table(m, "v2")
    if a.strict:
        s = json.loads(a.strict.read_text(encoding="utf-8"))
        lines += ["", "### Điểm vận hành “nghiêm” (FPR ≤ 1 % / ≤ 5 % trên dev, khóa trước test)", "",
                  "Ngưỡng: " + "; ".join(f"`{k}` low {f(v['low'], 4)} / high {f(v['high'], 4)}" for k, v in s["thresholds"].items()), ""] + \
                 product_table(s, "v2 nghiêm")
    if a.v1:
        v1 = json.loads(a.v1.read_text(encoding="utf-8"))
        lines += ["", "### TRƯỚC (v1: PSCC-Net, ngưỡng 0,35/0,65) trên cùng tập test", "", "#### Bài toán A của v1", ""] + A_tables(v1) + \
                 ["", "#### Cấp ứng dụng v1 (v1 chỉ có một bộ phát hiện)", ""] + product_table(v1, "v1")
    if a.ba:
        lines += ["", "### TRƯỚC / SAU trên cùng ảnh (ghép cặp)", ""] + ba_tables(json.loads(a.ba.read_text(encoding="utf-8")))
    if a.robust:
        r = json.loads(a.robust.read_text(encoding="utf-8"))
        lines += ["", "### Độ bền với hậu xử lý lành tính (cùng ảnh, cùng ngưỡng đã khóa)", ""] + robust_tables(r)
    a.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("written", a.out)


if __name__ == "__main__":
    main()
