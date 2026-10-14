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
    "not_applicable": "Không áp dụng",
    "unavailable": "Chưa khả dụng",
}


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

    def to_json(self) -> dict:
        return {"detector": self.key, "task": self.task, "task_vi": TASKS.get(self.task, self.task),
                "score": self.score, "score_type": self.score_type, "calibrated": self.calibrated,
                "decision": self.decision, "decision_vi": DECISIONS.get(self.decision, self.decision),
                "thresholds": self.thresholds, "has_heatmap": self.heatmap is not None,
                "heatmap_type": self.heatmap_type, "details": self.details, "warnings": self.warnings,
                "timing_seconds": self.timing}


def decide(score: float, low: float, high: float) -> str:
    if score >= high:
        return "suspicious"
    if score >= low:
        return "needs_review"
    return "no_clear_sign"
