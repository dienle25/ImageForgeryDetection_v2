"""Common detector contract.  Each detector answers ONE task and states its score semantics."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

TASKS = {
    "local_manipulation": "Chỉnh sửa cục bộ (ghép, sao chép vùng, xóa/thay thế)",
    "ai_generated": "Ảnh sinh hoàn toàn bằng AI",
    "face_deepfake": "Deepfake / hoán đổi khuôn mặt",
}

DECISIONS = {
    "no_clear_sign": "Chưa thấy dấu hiệu rõ",
    "needs_review": "Cần kiểm tra thêm",
    "suspicious": "Nghi vấn",
    # only the primary DefakerOne2 uses it: p_fake between its calibrated low and high thresholds
    "inconclusive": "Không kết luận",
    "not_applicable": "Không áp dụng",
    "unavailable": "Chưa khả dụng",
    "experimental": "Thử nghiệm — không tính vào kết luận",
}

# Add-ons reached over HTTP (DefakerOne2).  role "experimental" (default): shown next to the evaluated detectors
# but NEVER part of the overall decision, the task coverage or the decision-config hash.  role "primary": the
# overall decision follows this detector (the other detectors are shown for reference); if it is unavailable the
# decision falls back to the other detectors.  Their task is kept out of TASKS so the evaluated summary is unchanged.
EXPERIMENTAL_TYPES = {"defaker2_http"}
ROLES = ("experimental", "primary")
EXPERIMENTAL_TASKS = {
    "whole_image_authenticity": "Thật/giả toàn ảnh — DefakerOne2",
}


def task_label(task: str) -> str:
    return TASKS.get(task) or EXPERIMENTAL_TASKS.get(task, task)


def is_primary(det: dict) -> bool:
    return det.get("type") in EXPERIMENTAL_TYPES and det.get("role") == "primary"


def is_experimental(det: dict) -> bool:
    """An add-on that is shown only: never decides and never enters the decision-config hash."""
    return det.get("type") in EXPERIMENTAL_TYPES and not is_primary(det)


@dataclass
class DetectorResult:
    key: str
    task: str
    score: float | None
    score_type: str
    calibrated: bool
    decision: str
    thresholds: dict
    heatmap: np.ndarray | None = None
    heatmap_type: str | None = None
    details: dict = field(default_factory=dict)
    warnings: list = field(default_factory=list)
    timing: dict = field(default_factory=dict)
    in_summary: bool = True          # False = experimental: shown, but not counted in the overall decision
    primary: bool = False            # True = the overall decision follows this detector

    def to_json(self) -> dict:
        data = {"detector": self.key, "task": self.task, "task_vi": task_label(self.task),
                "score": self.score, "score_type": self.score_type, "calibrated": self.calibrated,
                "decision": self.decision, "decision_vi": DECISIONS.get(self.decision, self.decision),
                "thresholds": self.thresholds, "has_heatmap": self.heatmap is not None,
                "heatmap_type": self.heatmap_type, "details": self.details, "warnings": self.warnings,
                "timing_seconds": self.timing}
        if not self.in_summary:      # only written for add-on results: evaluated outputs stay byte-identical
            data["in_summary"] = False
        if self.primary:
            data["primary"] = True
        return data


def decide(score: float, low: float, high: float) -> str:
    if score >= high:
        return "suspicious"
    if score >= low:
        return "needs_review"
    return "no_clear_sign"
