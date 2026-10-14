"""Apply the PRE-REGISTERED default-on rule and quality targets (QUALITY_TARGETS.yaml) to the final-test metrics.

    python research/make_validation_summary.py --metrics reports/test/metrics_test_v2_standard.json \
        --config config.json --out-summary reports/validation_summary_vi.json --out-targets reports/test/targets_check.json

Written 2026-10-12 while the final test was still running and BEFORE any test metric was computed (EXPERIMENT_LOG P05).
The rule is applied mechanically; nothing here is tuned on test numbers.

Rule (QUALITY_TARGETS.yaml, selection_rules[2]): a detector stays ON by default only if the lower bound of the 95 %
bootstrap CI (resampling source groups) of its image-level AUROC on the independent final tests is > 0.60.
Operationalisation fixed here:
  * the primary final-test comparisons of each detector are listed in PRIMARY below (TASK_CONTRACT §5, P02);
  * every comparison passes  -> status "public_test_passed"  (no badge; text still says: not validated on company data)
  * some pass, some fail     -> status "experimental"        (badge THỬ NGHIỆM; text names where it failed)
  * none passes              -> status "disabled_by_rule"     (the detector must be switched off in config.json;
                                 product-level numbers are then recomputed from the stored scores without it)
The summary file is display-only: the app shows the text, it never changes scores, thresholds or decisions.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ifd.analysis import decision_config_hash  # noqa: E402
from ifd.config import load_config  # noqa: E402

RULE_CI_LOW = 0.60
PRIMARY = {
    "mesorch": [("local_manipulation", "ctrl_mei", "ảnh ghép/copy-move/xóa vật thể tổng hợp (ctrl_mei test)"),
                ("local_manipulation", "korus", "chỉnh sửa thủ công chuyên nghiệp (Korus test)"),
                ("local_manipulation", "cocoglide", "chỉnh sửa cục bộ bằng AI inpainting (CocoGlide)")],
    "commfor_aigen": [("ai_generated", "openfake_holdout_generators", "ảnh AI từ các generator của OpenFake test"),
                      ("ai_generated", "dff_text2img_vs_wiki", "ảnh chân dung tạo bằng Stable Diffusion (DeepFakeFace)")],
    "face_clip": [("face_deepfake", "faceswap_insightface_vs_real", "hoán đổi khuôn mặt InsightFace (DeepFakeFace)"),
                  ("face_deepfake", "face_inpainting_sd_vs_real", "vẽ lại khuôn mặt bằng SD inpainting (DeepFakeFace)")],
}
TASK_VI = {"mesorch": "chỉnh sửa cục bộ", "commfor_aigen": "ảnh tạo hoàn toàn bằng AI", "face_clip": "deepfake khuôn mặt"}


def f2(x) -> str:
    return "—" if x is None else f"{x:.2f}".replace(".", ",")


def pct(x) -> str:
    return "—" if x is None else f"{100 * x:.1f} %".replace(".", ",")


def comparisons(metrics: dict, det: str) -> list[dict]:
    out = []
    for task, key, label in PRIMARY[det]:
        blk = metrics.get("tasks", {}).get(task, {}).get(key)
        if not blk or blk.get("auroc") is None:
            out.append({"task": task, "set": key, "label_vi": label, "available": False, "passes": False})
            continue
        ci = blk.get("auroc_ci95") or [None, None]
        out.append({"task": task, "set": key, "label_vi": label, "available": True,
                    "n_pos": blk.get("n_pos"), "n_neg": blk.get("n_neg"), "auroc": blk["auroc"], "auroc_ci95": ci,
                    "tpr_at_high": (blk.get("at_high") or {}).get("recall"), "fpr_at_high": (blk.get("at_high") or {}).get("fpr"),
                    "passes": ci[0] is not None and ci[0] > RULE_CI_LOW})
    return out


def status_text(det: str, comps: list[dict]) -> tuple[str, str]:
    """Plain-language text for the UI: what share of each kind of fake reached “Nghi vấn”, plus AUROC."""
    passed = [c for c in comps if c["passes"]]
    failed = [c for c in comps if not c["passes"]]

    def item(c):
        if not c["available"]:
            return f"{c['label_vi']}: chưa có số liệu"
        return (f"{c['label_vi']}: {pct(c.get('tpr_at_high'))} ảnh giả/sửa và {pct(c.get('fpr_at_high'))} ảnh thật bị “Nghi vấn” "
                f"(AUROC {f2(c['auroc'])}, CI95 {f2(c['auroc_ci95'][0])}–{f2(c['auroc_ci95'][1])})")

    if not failed:
        return ("public_test_passed",
                "Đạt quy tắc bật mặc định trên dữ liệu kiểm thử công khai — " + "; ".join(item(c) for c in comps) +
                ". Chưa kiểm chứng trên ảnh của công ty; điểm không phải xác suất.")
    if passed:
        return ("experimental",
                "THỬ NGHIỆM — trên dữ liệu kiểm thử công khai: " + "; ".join(item(c) for c in comps) +
                ". Không đạt quy tắc bật mặc định (cận dưới CI95 AUROC ≤ 0,60) với: " +
                ", ".join(c["label_vi"] for c in failed) +
                ". “Chưa thấy dấu hiệu rõ” KHÔNG có nghĩa ảnh thật; cần người thẩm định.")
    return ("disabled_by_rule",
            "Không đạt quy tắc trên mọi tập final test (" + "; ".join(item(c) for c in comps) + "); tắt mặc định.")


def target_checks(metrics: dict, det_enabled: set[str]) -> list[dict]:
    T = metrics.get("tasks", {})
    rows = []

    def add(target, value, limit, kind, where):
        ok = None if value is None else (value >= limit if kind == "min" else value <= limit)
        rows.append({"target": target, "where": where, "value": value, "limit": limit, "kind": kind,
                     "met": ok})

    A = T.get("local_manipulation", {})
    for src in ("ctrl_mei", "korus", "cocoglide"):
        b = A.get(src)
        if b:
            add("local_manipulation.image_auroc_min", b.get("auroc"), 0.75, "min", src)
            add("local_manipulation.fpr_at_high_threshold_max", (b.get("at_high") or {}).get("fpr"), 0.10, "max", src)
    B = T.get("ai_generated", {})
    of = B.get("openfake_holdout_generators")
    if of:
        add("ai_generated.image_auroc_min_holdout_generators", of.get("auroc"), 0.85, "min", "openfake_test")
    for src, b in (B.get("false_positive_on_real_photos_by_source") or {}).items():
        add("ai_generated.fpr_on_camera_photos_at_high_threshold_max", b.get("fpr_at_high"), 0.05, "max", src)
    C = T.get("face_deepfake", {})
    for key in ("faceswap_insightface_vs_real", "face_inpainting_sd_vs_real"):
        b = C.get(key)
        if b:
            add("face_deepfake.image_auroc_min", b.get("auroc"), 0.80, "min", key)
    b = C.get("faceswap_insightface_vs_real")
    if b:
        add("face_deepfake.fpr_at_high_threshold_max", (b.get("at_high") or {}).get("fpr"), 0.10, "max", "dff_wiki_real_faces")
    for src, nb in (C.get("false_positive_on_other_real_images") or {}).items():
        add("face_deepfake.fpr_at_high_threshold_max", nb.get("fpr_at_high"), 0.10, "max", src + "_real_images")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--metrics", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--out-summary", type=Path, required=True)
    ap.add_argument("--out-targets", type=Path, required=True)
    ap.add_argument("--profile-name", default="chuẩn (FPR ≤ 5 % / ≤ 20 % trên dev)")
    ap.add_argument("--also-config", action="append", default=[], metavar="PATH=NAME",
                    help="another configuration whose numbers are ALSO in the quality report (e.g. the strict profile)")
    a = ap.parse_args()
    m = json.loads(a.metrics.read_text(encoding="utf-8"))
    cfg = json.loads(a.config.read_text(encoding="utf-8"))
    enabled = {k for k, v in cfg["detectors"].items() if v.get("enabled")}
    dets, detail = {}, {}
    for det in PRIMARY:
        if det not in enabled:
            continue
        comps = comparisons(m, det)
        status, text = status_text(det, comps)
        failed = [c["label_vi"] for c in comps if not c["passes"]]
        short = ("THỬ NGHIỆM — trên dữ liệu kiểm thử công khai không đạt với: " + ", ".join(failed) +
                 ". Chỉ dùng để sàng lọc; chi tiết ở mục “Phạm vi kiểm tra”." if status == "experimental" else
                 "Đạt kiểm thử công khai (chi tiết ở mục “Phạm vi kiểm tra”); chưa kiểm chứng trên ảnh của công ty."
                 if status == "public_test_passed" else text)
        dets[det] = {"status": status, "text_vi": text, "short_vi": short}
        detail[det] = {"status": status, "comparisons": comps}
    summary = {"schema": 1, "display_only": True, "rule": f"cận dưới CI95 AUROC > {RULE_CI_LOW} trên mọi tập final test chính",
               "source_metrics": a.metrics.as_posix(),
               # the app warns on every result when its decision-relevant configuration is none of these
               "evaluated_decision_configs": [{"name": a.profile_name, "sha256": decision_config_hash(load_config(a.config))}] +
                                             [{"name": spec.split("=", 1)[1], "sha256": decision_config_hash(load_config(Path(spec.split("=", 1)[0])))}
                                              for spec in a.also_config],
               "detectors": dets}
    a.out_summary.parent.mkdir(parents=True, exist_ok=True)
    a.out_summary.write_text(json.dumps(summary, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    targets = {"source_metrics": a.metrics.as_posix(), "rule_detail": detail, "targets": target_checks(m, enabled)}
    a.out_targets.parent.mkdir(parents=True, exist_ok=True)
    a.out_targets.write_text(json.dumps(targets, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    for det, d in dets.items():
        print(det, d["status"])
    print("targets met:", sum(1 for t in targets["targets"] if t["met"]), "/", len(targets["targets"]))


if __name__ == "__main__":
    main()
