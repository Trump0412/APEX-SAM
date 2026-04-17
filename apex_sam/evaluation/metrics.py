from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np
from scipy.ndimage import zoom

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional in lightweight test env
    cv2 = None


def compute_dice(pred: np.ndarray, gt: np.ndarray) -> float:
    if pred.shape != gt.shape:
        if cv2 is None:
            gt = zoom(gt.astype(np.float32), (pred.shape[0] / gt.shape[0], pred.shape[1] / gt.shape[1]), order=0)
        else:
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


def summarize_case_max_filtered(rows: Iterable[dict], threshold: float = 0.1) -> dict[str, object]:
    """Case-level summary:
    1) For each (label, case), keep the maximum slice Dice.
    2) Drop entries whose case Dice <= threshold.
    3) Average remaining entries.
    """
    rows = list(rows)
    case_best: dict[tuple[int, str], float] = {}
    for row in rows:
        key = (int(row["label"]), str(row["case_id"]))
        dice = float(row["dice"])
        if key not in case_best or dice > case_best[key]:
            case_best[key] = dice

    case_rows: list[dict[str, object]] = []
    by_label: dict[int, list[float]] = defaultdict(list)
    selected_all: list[float] = []
    threshold = float(threshold)

    for (label, case_id), case_dice in sorted(case_best.items()):
        keep = bool(case_dice > threshold)
        case_rows.append(
            {
                "label": int(label),
                "case_id": str(case_id),
                "case_dice": float(case_dice),
                "kept": int(keep),
            }
        )
        if keep:
            by_label[int(label)].append(float(case_dice))
            selected_all.append(float(case_dice))

    per_label = {label: float(np.mean(values)) for label, values in by_label.items()}
    overall = float(np.mean(selected_all)) if selected_all else 0.0

    return {
        "overall_mean_dice": overall,
        "per_label_mean_dice": per_label,
        "case_rows": case_rows,
        "num_case_entries": int(len(case_rows)),
        "num_case_entries_kept": int(len(selected_all)),
        "threshold": threshold,
    }
