"""Run the configured detectors on one image and assemble a self-describing report."""
from __future__ import annotations

import hashlib
import io
import json
import platform
import sys
import threading
import time
import zipfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import __version__
from .config import Config
from .detectors.base import DECISIONS, TASKS, DetectorResult
from .errors import ForgeryError
from .image_io import LoadedImage, decoded_pixel_hash, read_image
from .model_store import verify_model
from .render import binary_mask, colorize, draw_face_boxes, overlay, png_bytes

SCHEMA_VERSION = 2

SCOPE_NOTES = [
    "Điểm số là điểm thô của mô hình, KHÔNG phải xác suất đã hiệu chỉnh và không phải tỷ lệ phần trăm ảnh bị sửa.",
    "'Chưa thấy dấu hiệu rõ' không chứng minh ảnh là ảnh gốc; ảnh có thể đã mất dấu vết hoặc dùng kỹ thuật ngoài phạm vi.",
    "Bản đồ nhiệt là vùng mô hình kích hoạt, chưa phải vùng chỉnh sửa đã xác minh.",
    "Kết quả là gợi ý để người kiểm tra xem xét, không phải chứng cứ độc lập hay quyết định tự động.",
]


@dataclass
class Report:
    image: LoadedImage
    results: list
    summary: dict
    metadata: dict
    primary_heatmap: np.ndarray | None = None
    primary_heatmap_key: str | None = None
    pixel_threshold: float = 0.5

    def to_json(self) -> dict:
        data = dict(self.metadata)
        data["summary"] = self.summary
        data["detectors"] = [r.to_json() for r in self.results]
        return data


class Analyzer:
    """Loads every enabled detector once; analyze() is thread-safe."""

    def __init__(self, config: Config):
        self.config = config
        self.detectors = {}
        self.model_info = {}
        self._lock = threading.Lock()
        t0 = time.perf_counter()
        for key, det in config.data["detectors"].items():
            if not det.get("enabled"):
                continue
            model_path = config.path(det["model"])
            manifest = model_path.parent / "manifest.json"   # the manifest always sits next to the model files
            self.model_info[key] = verify_model(model_path, manifest)
            if det["type"] == "pscc_onnx":
                from .detectors.pscc import PSCCDetector
                self.detectors[key] = PSCCDetector(model_path, det, config.threads)
            elif det["type"] == "mesorch_onnx":
                from .detectors.mesorch import MesorchDetector
                self.detectors[key] = MesorchDetector(model_path, det, config.threads)
            elif det["type"] == "commfor_onnx":
                from .detectors.commfor import CommForDetector
                self.detectors[key] = CommForDetector(model_path, det, config.threads)
            elif det["type"] == "face_clip_onnx":
                from .detectors.face import FaceDeepfakeDetector
                face_path = config.path(det["face_model"])
                self.model_info[key + ":face_detector"] = verify_model(face_path, face_path.parent / "manifest.json")
                self.detectors[key] = FaceDeepfakeDetector(model_path, face_path, det, config.threads)
            else:
                raise ForgeryError(f"Loại detector không hỗ trợ: {det['type']}")
            # results always carry the config key, so several detectors of one class stay distinguishable
            self.detectors[key].key = key
        if not self.detectors:
            raise ForgeryError("Không có detector nào được bật trong config.json.")
        self.validation = load_validation_summary(config.base_dir)
        self.evaluated_configs = load_evaluated_configs(config.base_dir)
        self.load_seconds = time.perf_counter() - t0

    def config_matches_evaluated(self, cfg: Config | None = None) -> bool | None:
        """None = no evaluated configuration recorded; True/False = same/different decision-relevant settings."""
        if not self.evaluated_configs:
            return None
        return decision_config_hash(cfg or self.config) in self.evaluated_configs

    def evaluated_profile(self, cfg: Config | None = None) -> str | None:
        return self.evaluated_configs.get(decision_config_hash(cfg or self.config))

    def analyze(self, source, name: str | None = None, config: Config | None = None) -> Report:
        cfg = config or self.config
        lim = cfg.limits
        t0 = time.perf_counter()
        image = read_image(source, max_file_mb=lim["max_file_mb"], max_pixels=lim["max_pixels"],
                           min_side=lim["min_side"], name=name)
        read_s = time.perf_counter() - t0
        results: list[DetectorResult] = []
        for key, det in self.detectors.items():
            settings = cfg.detector(key)
            try:
                results.append(det.predict(image.rgb, settings))
            except ForgeryError:
                raise
            except MemoryError as exc:
                raise ForgeryError("Không đủ RAM để phân tích ảnh này; thử ảnh nhỏ hơn.") from exc
        for r in results:
            v = self.validation.get(r.key)
            if v:
                r.details["validation"] = v
        primary = next((r for r in results if r.heatmap is not None), None)
        summary = summarize(results, cfg)
        matches = self.config_matches_evaluated(cfg)
        if matches is False:
            summary["warnings"].insert(0, CONFIG_CHANGED_VI)
        metadata = {
            "schema_version": SCHEMA_VERSION, "app_version": __version__,
            "analyzed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "source_name": image.source_name, "image": image.metadata,
            "decoded_rgb_sha256": decoded_pixel_hash(image.rgb),
            "models": self.model_info, "config_sha256": config_hash(cfg), "config": cfg.to_json(),
            "decision_config_sha256": decision_config_hash(cfg), "config_matches_evaluated": matches,
            "evaluated_profile": self.evaluated_profile(cfg),
            "timing_seconds": {"read_decode": read_s, "total_processing": time.perf_counter() - t0,
                               "model_load": self.load_seconds},
            "environment": {"python": sys.version.split()[0], "platform": platform.platform(),
                            "onnxruntime": _ort_version(), "numpy": np.__version__, "device": "cpu"},
            "scope_notes": SCOPE_NOTES,
        }
        pixel_t = float(cfg.detector(primary.key)["thresholds"].get("pixel", 0.5)) if primary else 0.5
        return Report(image, results, summary, metadata,
                      primary.heatmap if primary else None, primary.key if primary else None, pixel_t)


APP_ROOT = Path(__file__).resolve().parent.parent


def _validation_summary(base_dir: Path) -> dict:
    """reports/validation_summary_vi.json next to the config file, else in the application root (the summary
    describes the shipped models, so alternative configs such as research/configs/config_strict.json use it too)."""
    for p in (Path(base_dir) / "reports" / "validation_summary_vi.json", APP_ROOT / "reports" / "validation_summary_vi.json"):
        try:
            data = json.loads(p.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except FileNotFoundError:
            continue
        except (OSError, ValueError):
            return {}
    return {}


def load_validation_summary(base_dir: Path) -> dict:
    """Display-only metadata written after the final test (status 'public_test_passed'/'experimental' + Vietnamese
    text).  It never changes scores, thresholds or decisions."""
    return {k: v for k, v in _validation_summary(base_dir).get("detectors", {}).items() if isinstance(v, dict)}


def _ort_version():
    try:
        import onnxruntime
        return onnxruntime.__version__
    except Exception:
        return None


def config_hash(cfg: Config) -> str:
    return hashlib.sha256(json.dumps(cfg.to_json(), sort_keys=True).encode()).hexdigest()


DECISION_KEYS = ("type", "thresholds", "image_score", "max_faces", "face_score_threshold", "policy")


def decision_config_hash(cfg: Config) -> str:
    """SHA-256 of everything in config.json that can change a score or a decision: which detectors are enabled,
    their type, thresholds (without the free-text 'status') and scoring settings.  Model files are pinned
    separately by SHA-256 (models/manifest.json); paths, limits, threads and display settings are excluded."""
    dets = {}
    for key, det in sorted(cfg.data["detectors"].items()):
        if not det.get("enabled"):
            continue
        d = {k: det[k] for k in DECISION_KEYS if k in det}
        d["thresholds"] = {k: v for k, v in det.get("thresholds", {}).items() if k != "status"}
        dets[key] = d
    return hashlib.sha256(json.dumps(dets, sort_keys=True).encode()).hexdigest()


def load_evaluated_configs(base_dir: Path) -> dict[str, str]:
    """{decision_config_sha256: profile name} of the configurations whose numbers are in the quality report."""
    data = _validation_summary(base_dir)
    out = {}
    for item in data.get("evaluated_decision_configs", []) or []:
        if isinstance(item, dict) and isinstance(item.get("sha256"), str) and len(item["sha256"]) == 64:
            out[item["sha256"]] = str(item.get("name", ""))
    legacy = data.get("evaluated_decision_config_sha256")
    if isinstance(legacy, str) and len(legacy) == 64:
        out.setdefault(legacy, "")
    return out


CONFIG_CHANGED_VI = ("Cấu hình bộ phát hiện/ngưỡng KHÁC với cấu hình đã được đánh giá trong báo cáo chất lượng; "
                     "các số liệu chất lượng đã báo cáo không còn áp dụng cho kết quả này.")


def summarize(results: list[DetectorResult], cfg: Config) -> dict:
    order = {"suspicious": 3, "needs_review": 2, "no_clear_sign": 1}
    applicable = [r for r in results if r.decision in order]
    worst = max((r.decision for r in applicable), key=lambda d: order[d], default="no_clear_sign")
    covered = sorted({r.task for r in applicable})
    not_applicable = sorted({r.task for r in results if r.decision == "not_applicable"} - set(covered))
    not_covered = [t for t in TASKS if t not in covered and t not in not_applicable]
    headline = {
        "suspicious": "Có dấu hiệu nghi vấn — cần người kiểm tra xem xét",
        "needs_review": "Cần kiểm tra thêm — tín hiệu chưa rõ ràng",
        "no_clear_sign": "Chưa thấy dấu hiệu rõ trong phạm vi đã kiểm tra",
    }[worst]
    warnings = []
    for r in results:
        if r.heatmap is not None and r.decision == "no_clear_sign":
            frac = r.details.get("heatmap_stats", {}).get("fraction_ge_pixel_threshold", 0)
            if frac >= 0.5:
                warnings.append("Bản đồ nhiệt có ≥ 50 % diện tích ở mức ≥ ngưỡng pixel nhưng điểm chưa tới ngưỡng kết luận: "
                                "nên xem kỹ ảnh (quy tắc mô tả, chưa hiệu chỉnh; không thay đổi kết luận).")
        warnings.extend(r.warnings)
    return {"overall_decision": worst, "overall_decision_vi": DECISIONS[worst], "headline_vi": headline,
            "tasks_checked": [{"task": t, "task_vi": TASKS[t]} for t in covered],
            "tasks_not_applicable": [{"task": t, "task_vi": TASKS[t],
                                      "reason_vi": "Không phát hiện khuôn mặt trong ảnh" if t == "face_deepfake" else "Không áp dụng"}
                                     for t in not_applicable],
            "tasks_not_checked": [{"task": t, "task_vi": TASKS[t]} for t in not_covered],
            "warnings": warnings}


def report_files(report: Report, alpha: float, arrays: dict | None = None, images: bool = True) -> dict[str, bytes]:
    """All downloadable files.  If `arrays` is a dict, the rendered full-resolution heatmap/overlay/faces arrays are
    stored in it so that callers (web previews) do not render them a second time.  images=False: only result.json
    (scores, decisions, metadata) — used by the CLI option --no-images for fast batch screening."""
    data = report.to_json()
    if not images:
        data["artifacts"] = ["result.json"]
        return {"result.json": (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")}
    files: dict[str, bytes] = {"original.png": png_bytes(report.image.rgb)}
    if report.primary_heatmap is not None:
        hm = report.primary_heatmap
        colored = colorize(hm)
        over = overlay(report.image.rgb, hm, alpha, colored=colored)
        if arrays is not None:
            arrays["heatmap"], arrays["overlay"] = colored, over
        files["heatmap.png"] = png_bytes(colored)
        files["overlay.png"] = png_bytes(over)
        files["mask.png"] = png_bytes(binary_mask(hm, report.pixel_threshold))
        files["heatmap_gray16.png"] = png_bytes(np.round(hm * 65535).astype(np.uint16))
        buf = io.BytesIO()
        np.save(buf, hm.astype(np.float32), allow_pickle=False)
        files["heatmap.npy"] = buf.getvalue()
        data["visualization"] = {"heatmap_detector": report.primary_heatmap_key, "color_scale": "cố định [0,1]: xanh→cyan→vàng→đỏ",
                                 "overlay_opacity": "alpha × điểm pixel", "overlay_alpha": alpha,
                                 "mask_threshold": report.pixel_threshold}
    face_res = next((r for r in report.results if r.task == "face_deepfake" and r.details.get("faces")), None)
    if face_res is not None:
        boxes = draw_face_boxes(report.image.rgb, face_res.details["faces"])
        if arrays is not None:
            arrays["faces"] = boxes
        files["faces.png"] = png_bytes(boxes)
    data["artifacts"] = sorted(list(files) + ["result.json"])
    files["result.json"] = (json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n").encode("utf-8")
    return files


def zip_bytes(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for name, data in files.items():
            zf.writestr(name, data)
    return buf.getvalue()
