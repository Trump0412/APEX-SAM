from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class SupportMeta:
    support_id: str = ""
    case_id: str = ""
    slice_path: str = ""
    label_path: str = ""
    slice_index: int = -1
    label_value: int | None = None


@dataclass
class SupportMatch:
    score: float
    meta: SupportMeta
    image: np.ndarray | None = None
    mask: np.ndarray | None = None


@dataclass
class PredictionResult:
    pred_mask: np.ndarray
    pre_mask: np.ndarray | None
    points_pos: np.ndarray
    points_neg: np.ndarray
    bbox: tuple[int, int, int, int] | None
    debug_paths: list[str] = field(default_factory=list)
    debug: dict[str, Any] = field(default_factory=dict)


@dataclass
class RunSummary:
    run_dir: str
    num_cases: int
    num_slices: int
    mean_dice: float
    metrics_csv: str
    summary_json: str
