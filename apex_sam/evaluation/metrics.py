from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import cv2
import numpy as np


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    if pred.shape != gt.shape:
        gt = cv2.resize(gt.astype(np.float32), (pred.shape[1], pred.shape[0]), interpolation=cv2.INTER_NEAREST)
    pred_bin = (pred > 0.5).astype(np.float32)
    gt_bin = (gt > 0.5).astype(np.float32)
    inter = float((pred_bin * gt_bin).sum())
    union = float(pred_bin.sum() + gt_bin.sum())
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return float(2.0 * inter / union)


def summarize_by_label(rows: Iterable[dict]) -> dict[str, float | dict[int, float]]:
    rows = list(rows)
    by_label = defaultdict(list)
    for row in rows:
        by_label[int(row['label'])].append(float(row['dice']))
    per_label = {label: float(np.mean(values)) for label, values in by_label.items()}
    overall = float(np.mean([float(row['dice']) for row in rows])) if rows else 0.0
    return {'overall_mean_dice': overall, 'per_label_mean_dice': per_label}
