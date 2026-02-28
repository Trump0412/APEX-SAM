from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np


def create_run_dir(output_root: str) -> str:
    run_dir = Path(output_root) / f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / 'preds').mkdir(exist_ok=True)
    (run_dir / 'overlays').mkdir(exist_ok=True)
    return str(run_dir)


def save_metrics_csv(rows: Iterable[dict], path: str) -> str:
    rows = list(rows)
    fieldnames = ['label', 'case_id', 'slice_id', 'dice', 'support_case', 'support_slice', 'support_score', 'pred_path']
    with open(path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k) for k in fieldnames})
    return path


def save_summary_json(summary: dict, path: str) -> str:
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2)
    return path


def save_overlay(image: np.ndarray, pred_mask: np.ndarray, gt_mask: np.ndarray, path: str) -> str:
    base = image.astype(np.float32)
    if base.ndim == 2:
        base = base - base.min()
        base = base / (base.max() - base.min() + 1e-8)
        base = (base * 255).astype(np.uint8)
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)
    pred = pred_mask > 0.5
    gt = gt_mask > 0.5
    tp = pred & gt
    fp = pred & ~gt
    fn = ~pred & gt
    out = base.copy()
    out[tp] = (0, 255, 0)
    out[fp] = (0, 0, 255)
    out[fn] = (255, 0, 0)
    cv2.imwrite(path, out)
    return path
