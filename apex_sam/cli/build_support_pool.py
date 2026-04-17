from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from apex_sam.data.normalized import case_id_from_path, iter_cases, load_case

try:
    import cv2  # type: ignore
except Exception:  # pragma: no cover - optional for CI without OpenCV
    cv2 = None


def _evenly_sample_indices(length: int, sample_count: int) -> list[int]:
    if length <= 0:
        return []
    if sample_count >= length:
        return list(range(length))
    picked = np.linspace(0, length - 1, num=sample_count, dtype=np.int64)
    uniq = sorted(set(int(i) for i in picked.tolist()))
    if len(uniq) < sample_count:
        for idx in range(length):
            if idx not in uniq:
                uniq.append(idx)
            if len(uniq) >= sample_count:
                break
    return sorted(uniq[:sample_count])


def _preview_uint8(image: np.ndarray) -> np.ndarray:
    arr = np.asarray(image, dtype=np.float32)
    lo = float(np.percentile(arr, 1))
    hi = float(np.percentile(arr, 99))
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    norm = np.clip((arr - lo) / (hi - lo), 0.0, 1.0)
    return (norm * 255.0).astype(np.uint8)


def build_support_pool(
    data_dir: str,
    output_dir: str,
    dataset: str,
    labels: list[int],
    max_support_per_label: int,
    min_mask_area: int,
) -> dict[str, Any]:
    output_root = Path(output_dir).expanduser().resolve()
    slices_root = output_root / "support_slices"
    manifest_root = output_root / "manifest"
    slices_root.mkdir(parents=True, exist_ok=True)
    manifest_root.mkdir(parents=True, exist_ok=True)

    volume_cache: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    selected: dict[tuple[str, int], dict[str, Any]] = {}
    stats_per_label: dict[int, int] = {}

    pairs = iter_cases(data_dir)
    for label_value in labels:
        candidates: list[dict[str, Any]] = []
        for image_path, label_path in pairs:
            cache_key = f"{image_path}::{label_path}"
            if cache_key not in volume_cache:
                volume_cache[cache_key] = load_case(image_path, label_path, dataset=dataset)
            image_volume, label_volume = volume_cache[cache_key]
            case_id = case_id_from_path(image_path)
            valid_slices = np.where((label_volume == int(label_value)).sum(axis=(1, 2)) >= int(min_mask_area))[0]
            for slice_index in valid_slices.tolist():
                img_slice = np.asarray(image_volume[int(slice_index)], dtype=np.float32)
                mask_slice = (label_volume[int(slice_index)] == int(label_value)).astype(np.uint8)
                if mask_slice.sum() < int(min_mask_area):
                    continue
                present_labels = sorted(int(v) for v in np.unique(label_volume[int(slice_index)]) if int(v) > 0)
                candidates.append(
                    {
                        "case_id": case_id,
                        "slice_index": int(slice_index),
                        "image": img_slice,
                        "mask": mask_slice,
                        "label": int(label_value),
                        "present_labels": present_labels,
                    }
                )
        if not candidates:
            stats_per_label[int(label_value)] = 0
            continue
        pick_count = min(int(max_support_per_label), len(candidates))
        pick_indices = _evenly_sample_indices(len(candidates), pick_count)
        stats_per_label[int(label_value)] = len(pick_indices)
        for index in pick_indices:
            item = candidates[int(index)]
            key = (item["case_id"], int(item["slice_index"]))
            entry = selected.setdefault(
                key,
                {
                    "case_id": item["case_id"],
                    "slice_index": int(item["slice_index"]),
                    "image": item["image"],
                    "present_labels": set(item["present_labels"]),
                    "selected_for_labels": set(),
                    "masks": {},
                },
            )
            entry["present_labels"].update(item["present_labels"])
            entry["selected_for_labels"].add(int(label_value))
            entry["masks"][int(label_value)] = item["mask"]

    support_rows: list[dict[str, Any]] = []
    for case_id, slice_index in sorted(selected.keys(), key=lambda x: (x[0], x[1])):
        entry = selected[(case_id, slice_index)]
        folder = slices_root / f"case_{case_id}_slice_{int(slice_index):03d}"
        folder.mkdir(parents=True, exist_ok=True)
        image = np.asarray(entry["image"], dtype=np.float32)
        np.save(folder / "image.npy", image)
        if cv2 is not None:
            cv2.imwrite(str(folder / "image.png"), _preview_uint8(image))

        for label_value, mask in sorted(entry["masks"].items()):
            mask_u8 = (np.asarray(mask) > 0).astype(np.uint8)
            np.save(folder / f"mask_label{int(label_value)}.npy", mask_u8)
            if cv2 is not None:
                cv2.imwrite(str(folder / f"mask_label{int(label_value)}.png"), mask_u8 * 255)

        meta = {
            "case_id": case_id,
            "slice_index": int(slice_index),
            "present_labels": sorted(int(v) for v in entry["present_labels"]),
            "selected_for_query_labels": sorted(int(v) for v in entry["selected_for_labels"]),
            "dataset": dataset,
        }
        (folder / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        support_rows.append(
            {
                "support_id": folder.name,
                "case_id": case_id,
                "slice_index": int(slice_index),
                "present_labels": " ".join(str(v) for v in meta["present_labels"]),
                "selected_for_query_labels": " ".join(str(v) for v in meta["selected_for_query_labels"]),
            }
        )

    summary = {
        "dataset": dataset,
        "data_dir": str(Path(data_dir).expanduser().resolve()),
        "labels": [int(v) for v in labels],
        "max_support_per_label": int(max_support_per_label),
        "min_mask_area": int(min_mask_area),
        "num_support_slices": len(support_rows),
        "stats_per_label": stats_per_label,
        "support_pool_dir": str(output_root),
    }

    with open(manifest_root / "support_summary.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["support_id", "case_id", "slice_index", "present_labels", "selected_for_query_labels"],
        )
        writer.writeheader()
        writer.writerows(support_rows)
    (manifest_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a public support pool from a normalized dataset.")
    parser.add_argument("--dataset", default="CHAOS_MR_T2", choices=["CHAOS_MR_T2", "CHAOS_CT", "MSCMR", "MS-CMR", "SATA_CAP"])
    parser.add_argument("--data-dir", required=True, help="Dataset root that contains normalized/image_*.nii.gz and label_*.nii.gz")
    parser.add_argument("--output-dir", required=True, help="Output support pool directory")
    parser.add_argument("--labels", type=int, nargs="*", default=[1, 2, 3, 4], help="Labels to include in the support pool")
    parser.add_argument("--max-support-per-label", type=int, default=24)
    parser.add_argument("--min-mask-area", type=int, default=32)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = build_support_pool(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        dataset=args.dataset,
        labels=list(args.labels),
        max_support_per_label=args.max_support_per_label,
        min_mask_area=args.min_mask_area,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
