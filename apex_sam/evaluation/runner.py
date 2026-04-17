from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from apex_sam.config import ApexConfig
from apex_sam.data.io import resize_image_2d, resize_mask_2d
from apex_sam.data.normalized import case_id_from_path, iter_cases, iter_label_slices, load_case
from apex_sam.evaluation.metrics import compute_dice, summarize_by_label
from apex_sam.evaluation.reporting import create_run_dir, save_metrics_csv, save_overlay, save_summary_json
from apex_sam.pipeline.segmenter import ApexSegmenter
from apex_sam.retrieval.local_db import LocalSupportDB
from apex_sam.support.filesystem_pool import FileSystemSupportPool
from apex_sam.types import RunSummary, SupportMatch


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


def _legacy_load_support_for_label(match: SupportMatch, label_value: int, dataset: str):
    image_volume, label_volume = load_case(match.meta.slice_path, match.meta.label_path, dataset=dataset)
    image_slice = image_volume[int(match.meta.slice_index)]
    mask_slice = (label_volume[int(match.meta.slice_index)] == int(label_value)).astype(np.uint8)
    return image_slice, mask_slice


def run_evaluation(config: ApexConfig) -> RunSummary:
    run_dir = create_run_dir(config.output_root)
    logger = _setup_logger(run_dir)
    logger.info("Run directory: %s", run_dir)

    segmenter = ApexSegmenter(config)

    support_provider = None
    local_db = None
    support_source = ""
    if config.support_pool_dir:
        support_provider = FileSystemSupportPool(config.support_pool_dir, encoder=segmenter.dino_encoder)
        support_source = f"support_pool:{config.support_pool_dir}"
        logger.info("Support provider: filesystem pool (%s slices)", len(support_provider.entries))
    elif config.local_db_path:
        local_db = LocalSupportDB.load(config.local_db_path, encoder=segmenter.dino_encoder)
        support_source = f"legacy_local_db:{config.local_db_path}"
        logger.warning(
            "Falling back to legacy --local-db-path flow. "
            "For the open-source release, prefer --support-pool-dir."
        )
    else:
        raise RuntimeError("Either --support-pool-dir or --local-db-path must be provided.")

    records: list[dict[str, Any]] = []
    seen_cases = 0
    seen_slices = 0

    for label_value in config.test_labels:
        pred_dir_label = Path(run_dir) / "preds" / f"label{int(label_value)}"
        overlay_dir_label = Path(run_dir) / "overlays" / f"label{int(label_value)}"
        pred_dir_label.mkdir(parents=True, exist_ok=True)
        overlay_dir_label.mkdir(parents=True, exist_ok=True)

        for case_index, (image_path, label_path) in enumerate(iter_cases(config.data_dir)):
            if config.max_cases is not None and case_index >= int(config.max_cases):
                break

            case_id = case_id_from_path(image_path)
            image_volume, label_volume = load_case(image_path, label_path, dataset=config.dataset)
            valid_slices = iter_label_slices(label_volume, int(label_value), config.max_slices)
            if len(valid_slices) == 0:
                continue
            seen_cases += 1

            for slice_id in valid_slices:
                query_image = image_volume[int(slice_id)]
                query_mask = (label_volume[int(slice_id)] == int(label_value)).astype(np.float32)

                if support_provider is not None:
                    results = support_provider.search(
                        query_image,
                        label_value=int(label_value),
                        topk=int(config.retrieval_topk),
                        exclude_case_id=case_id if config.retrieval_skip_self else None,
                        exclude_slice_index=int(slice_id) if config.retrieval_skip_self else None,
                    )
                    valid_candidates = [r for r in results if r.image is not None and r.mask is not None and np.asarray(r.mask).sum() > 0]
                    if not valid_candidates:
                        logger.warning("[Case %s][Slice %s] No valid support with label=%s.", case_id, slice_id, label_value)
                        continue
                    pick_idx = min(max(1, int(config.retrieval_rank)) - 1, len(valid_candidates) - 1)
                    match = valid_candidates[pick_idx]
                    support_image = np.asarray(match.image, dtype=np.float32)
                    support_mask = np.asarray(match.mask, dtype=np.uint8)
                else:
                    assert local_db is not None
                    results = local_db.search(query_image, topk=config.retrieval_topk)
                    valid_candidates_all = []
                    valid_candidates = []
                    for match_item in results:
                        support_image_i, support_mask_i = _legacy_load_support_for_label(match_item, int(label_value), dataset=config.dataset)
                        if np.asarray(support_mask_i).sum() <= 0:
                            continue
                        entry = {"match": match_item, "image": support_image_i, "mask": support_mask_i}
                        valid_candidates_all.append(entry)
                        support_case = case_id_from_path(match_item.meta.slice_path)
                        if config.retrieval_skip_self and support_case == case_id and int(match_item.meta.slice_index) == int(slice_id):
                            continue
                        valid_candidates.append(entry)
                    if not valid_candidates:
                        if valid_candidates_all:
                            logger.warning(
                                "[Case %s][Slice %s] Only self-matching supports found; falling back to allow self.",
                                case_id,
                                slice_id,
                            )
                            valid_candidates = valid_candidates_all
                        else:
                            logger.warning("[Case %s][Slice %s] No valid support with label=%s.", case_id, slice_id, label_value)
                            continue
                    pick_idx = min(max(1, int(config.retrieval_rank)) - 1, len(valid_candidates) - 1)
                    chosen = valid_candidates[pick_idx]
                    support_image = chosen["image"]
                    support_mask = chosen["mask"]
                    match = chosen["match"]

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
                pred_path = pred_dir_label / f"{case_id}_slice{int(slice_id)}_label{int(label_value)}_pred.npy"
                np.save(pred_path, pred_mask.astype(np.float32))

                overlay_path = overlay_dir_label / f"{case_id}_slice{int(slice_id)}_label{int(label_value)}.png"
                save_overlay(query_image, pred_mask, query_mask, str(overlay_path))

                dice = compute_dice(pred_mask, query_mask)
                support_case = getattr(match.meta, "case_id", "")
                if not support_case:
                    support_case = case_id_from_path(match.meta.slice_path) if match.meta.slice_path else ""
                support_slice = int(match.meta.slice_index) if getattr(match.meta, "slice_index", None) is not None else -1

                logger.info("label=%s case=%s slice=%s dice=%.4f", int(label_value), case_id, int(slice_id), float(dice))
                records.append(
                    {
                        "label": int(label_value),
                        "case_id": case_id,
                        "slice_id": int(slice_id),
                        "dice": float(dice),
                        "support_case": support_case,
                        "support_slice": support_slice,
                        "support_score": float(match.score),
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
        "support_source": support_source,
        "labels": list(config.test_labels),
        "mean_dice_overall": dice_summary["overall_mean_dice"],
        "mean_dice_per_label": dice_summary["per_label_mean_dice"],
        "num_cases": seen_cases,
        "num_slices": seen_slices,
        "run_dir": run_dir,
        "metrics_csv": metrics_csv,
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
