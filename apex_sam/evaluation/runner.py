from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from apex_sam.config import ApexConfig
from apex_sam.data.io import load_nifti, resize_image_2d, resize_mask_2d
from apex_sam.data.normalized import case_id_from_path, iter_cases, iter_label_slices, load_case
from apex_sam.evaluation.metrics import compute_dice, summarize_by_label
from apex_sam.evaluation.reporting import create_run_dir, save_metrics_csv, save_overlay, save_summary_json
from apex_sam.pipeline.segmenter import ApexSegmenter
from apex_sam.types import RunSummary


def _setup_logger(run_dir: str) -> logging.Logger:
    logger = logging.getLogger(f"apex_sam_run_{Path(run_dir).name}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(name)s - %(message)s")
    fh = logging.FileHandler(Path(run_dir) / "run.log")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)
    return logger


def _load_array(path: str) -> np.ndarray:
    lower = path.lower()
    if lower.endswith(".npy"):
        return np.load(path)
    if lower.endswith(".nii") or lower.endswith(".nii.gz"):
        return load_nifti(path)
    raise RuntimeError(f"Unsupported file format for support input: {path}")


def _to_2d(arr: np.ndarray) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        return arr[arr.shape[0] // 2]
    raise RuntimeError(f"Expected 2D/3D array, got shape={arr.shape}")


def _resolve_support_pair(config: ApexConfig, label_value: int) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    label_value = int(label_value)

    if config.support_item_dir:
        support_dir = Path(config.support_item_dir).expanduser().resolve()
        image_path = support_dir / "image.npy"
        mask_path = support_dir / f"mask_label{label_value}.npy"
        if not image_path.exists():
            raise RuntimeError(f"Missing support image in support_item_dir: {image_path}")
        if not mask_path.exists():
            raise RuntimeError(f"Missing support mask for label={label_value}: {mask_path}")
        support_image = _to_2d(_load_array(str(image_path))).astype(np.float32)
        support_mask = (_to_2d(_load_array(str(mask_path))) > 0.5).astype(np.uint8)
        meta = {
            "support_mode": "support_item_dir",
            "support_image_path": str(image_path),
            "support_mask_path": str(mask_path),
        }
        return support_image, support_mask, meta

    if not config.support_image_path:
        raise RuntimeError("Please provide --support-item-dir or --support-image-path.")

    support_image = _to_2d(_load_array(config.support_image_path)).astype(np.float32)
    if config.support_mask_template:
        mask_path = config.support_mask_template.format(label=label_value)
    else:
        mask_path = config.support_mask_path
    if not mask_path:
        raise RuntimeError("Please provide --support-mask-path or --support-mask-template.")

    support_mask = (_to_2d(_load_array(mask_path)) > 0.5).astype(np.uint8)
    meta = {
        "support_mode": "explicit_paths",
        "support_image_path": str(config.support_image_path),
        "support_mask_path": str(mask_path),
    }
    return support_image, support_mask, meta


def run_evaluation(config: ApexConfig) -> RunSummary:
    run_dir = create_run_dir(config.output_root)
    logger = _setup_logger(run_dir)
    logger.info("Run directory: %s", run_dir)

    if config.expert_database_dir:
        logger.info("Expert database path (Module-1 placeholder): %s", config.expert_database_dir)

    segmenter = ApexSegmenter(config)
    records: list[dict[str, Any]] = []
    seen_cases = 0
    seen_slices = 0

    label_support_cache: dict[int, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}

    for label_value in config.test_labels:
        label_value = int(label_value)
        support_image, support_mask, support_meta = _resolve_support_pair(config, label_value)
        label_support_cache[label_value] = (support_image, support_mask, support_meta)

        pred_dir_label = Path(run_dir) / "preds" / f"label{label_value}"
        overlay_dir_label = Path(run_dir) / "overlays" / f"label{label_value}"
        pred_dir_label.mkdir(parents=True, exist_ok=True)
        overlay_dir_label.mkdir(parents=True, exist_ok=True)

        for case_index, (image_path, label_path) in enumerate(iter_cases(config.data_dir)):
            if config.max_cases is not None and case_index >= int(config.max_cases):
                break

            case_id = case_id_from_path(image_path)
            image_volume, label_volume = load_case(image_path, label_path, dataset=config.dataset)
            valid_slices = iter_label_slices(label_volume, label_value, config.max_slices)
            if len(valid_slices) == 0:
                continue
            seen_cases += 1

            for slice_id in valid_slices:
                query_image = image_volume[int(slice_id)]
                query_mask = (label_volume[int(slice_id)] == label_value).astype(np.float32)

                query_image_in = resize_image_2d(query_image, (config.force_input_size, config.force_input_size))
                query_mask_in = resize_mask_2d(query_mask, (config.force_input_size, config.force_input_size))
                support_image_in = resize_image_2d(support_image, (config.force_input_size, config.force_input_size))
                support_mask_in = resize_mask_2d(support_mask, (config.force_input_size, config.force_input_size))

                result = segmenter.predict(
                    support_image_in,
                    support_mask_in,
                    query_image_in,
                    case_id=case_id,
                    slice_id=int(slice_id),
                    gt_mask=query_mask_in,
                    viz_dir=None,
                    logger=None,
                )
                pred_mask = resize_mask_2d(result.pred_mask.astype(np.float32), query_mask.shape)
                pred_path = pred_dir_label / f"{case_id}_slice{int(slice_id)}_label{label_value}_pred.npy"
                np.save(pred_path, pred_mask.astype(np.float32))

                overlay_path = overlay_dir_label / f"{case_id}_slice{int(slice_id)}_label{label_value}.png"
                save_overlay(query_image, pred_mask, query_mask, str(overlay_path))

                dice = compute_dice(pred_mask, query_mask)
                logger.info("label=%s case=%s slice=%s dice=%.4f", label_value, case_id, int(slice_id), float(dice))

                records.append(
                    {
                        "label": label_value,
                        "case_id": case_id,
                        "slice_id": int(slice_id),
                        "dice": float(dice),
                        "support_case": support_meta.get("support_image_path", ""),
                        "support_slice": -1,
                        "support_score": 0.0,
                        "pred_path": str(pred_path),
                    }
                )
                seen_slices += 1

    metrics_csv = save_metrics_csv(records, str(Path(run_dir) / "metrics.csv"))
    dice_summary = summarize_by_label(records)
    summary = {
        "config": config.public_dict(),
        "data_dir": config.data_dir,
        "dataset": config.dataset,
        "expert_database_dir": config.expert_database_dir,
        "labels": list(config.test_labels),
        "mean_dice_overall": dice_summary["overall_mean_dice"],
        "mean_dice_per_label": dice_summary["per_label_mean_dice"],
        "num_cases": seen_cases,
        "num_slices": seen_slices,
        "run_dir": run_dir,
        "metrics_csv": metrics_csv,
        "support_by_label": {
            str(k): {
                "support_mode": v[2].get("support_mode", ""),
                "support_image_path": v[2].get("support_image_path", ""),
                "support_mask_path": v[2].get("support_mask_path", ""),
            }
            for k, v in label_support_cache.items()
        },
    }
    summary_json = save_summary_json(summary, str(Path(run_dir) / "summary.json"))
    return RunSummary(
        run_dir=run_dir,
        num_cases=seen_cases,
        num_slices=seen_slices,
        mean_dice=float(dice_summary["overall_mean_dice"]),
        metrics_csv=metrics_csv,
        summary_json=summary_json,
    )
