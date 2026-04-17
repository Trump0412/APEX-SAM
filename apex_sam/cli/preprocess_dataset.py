from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from apex_sam.data.io import remap_labels

try:
    import SimpleITK as sitk  # type: ignore
except Exception:  # pragma: no cover
    sitk = None


def _extract_key(path: Path) -> str:
    stem = path.name
    if stem.endswith(".nii.gz"):
        stem = stem[:-7]
    else:
        stem = path.stem
    numbers = re.findall(r"\d+", stem)
    if numbers:
        return numbers[-1]
    return stem


def _normalize_volume(image_arr: np.ndarray, p1: float, p99: float) -> np.ndarray:
    image_arr = np.asarray(image_arr, dtype=np.float32)
    lo, hi = np.percentile(image_arr, [p1, p99])
    if hi <= lo:
        return np.zeros_like(image_arr, dtype=np.float32)
    clipped = np.clip(image_arr, lo, hi)
    return ((clipped - lo) / (hi - lo)).astype(np.float32)


def _save_array_like(reference: sitk.Image, array: np.ndarray, path: Path, is_label: bool) -> None:
    arr = np.asarray(array)
    if is_label:
        arr = arr.astype(np.uint8)
    else:
        arr = arr.astype(np.float32)
    image = sitk.GetImageFromArray(arr)
    image.CopyInformation(reference)
    sitk.WriteImage(image, str(path))


def preprocess_dataset(
    dataset: str,
    image_dir: str,
    label_dir: str,
    output_dir: str,
    image_glob: str = "*.nii.gz",
    label_glob: str = "*.nii.gz",
    p1: float = 1.0,
    p99: float = 99.0,
) -> dict:
    if sitk is None:
        raise RuntimeError("SimpleITK is required for dataset preprocessing.")
    image_root = Path(image_dir).expanduser().resolve()
    label_root = Path(label_dir).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    normalized_root = output_root / "normalized"
    normalized_root.mkdir(parents=True, exist_ok=True)

    image_files = sorted(image_root.glob(image_glob))
    label_files = sorted(label_root.glob(label_glob))
    if not image_files:
        raise RuntimeError(f"No image files found: {image_root}/{image_glob}")
    if not label_files:
        raise RuntimeError(f"No label files found: {label_root}/{label_glob}")

    label_map = {_extract_key(path): path for path in label_files}
    pairs: list[tuple[Path, Path]] = []
    missing: list[str] = []
    for image_path in image_files:
        key = _extract_key(image_path)
        label_path = label_map.get(key)
        if label_path is None:
            missing.append(str(image_path))
            continue
        pairs.append((image_path, label_path))
    if missing:
        raise RuntimeError(f"Missing labels for {len(missing)} image files. Example: {missing[:3]}")
    if not pairs:
        raise RuntimeError("No image/label pairs found.")

    width = max(3, len(str(len(pairs) - 1)))
    rows = []
    for index, (image_path, label_path) in enumerate(pairs):
        image_itk = sitk.ReadImage(str(image_path))
        label_itk = sitk.ReadImage(str(label_path))
        image_arr = sitk.GetArrayFromImage(image_itk).astype(np.float32)
        label_arr = sitk.GetArrayFromImage(label_itk)

        image_norm = _normalize_volume(image_arr, p1=p1, p99=p99)
        label_norm = remap_labels(label_arr, dataset=dataset)

        stem = f"{index:0{width}d}"
        image_out = normalized_root / f"image_{stem}.nii.gz"
        label_out = normalized_root / f"label_{stem}.nii.gz"
        _save_array_like(image_itk, image_norm, image_out, is_label=False)
        _save_array_like(label_itk, label_norm, label_out, is_label=True)

        rows.append(
            {
                "index": int(index),
                "id": stem,
                "image_in": str(image_path),
                "label_in": str(label_path),
                "image_out": str(image_out),
                "label_out": str(label_out),
            }
        )

    summary = {
        "dataset": dataset,
        "num_cases": len(rows),
        "output_dir": str(output_root),
        "normalized_dir": str(normalized_root),
        "clip_percentiles": [float(p1), float(p99)],
    }
    (output_root / "preprocess_manifest.json").write_text(
        json.dumps({"summary": summary, "cases": rows}, indent=2),
        encoding="utf-8",
    )
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Preprocess a medical dataset into the normalized APEX-SAM format.")
    parser.add_argument("--dataset", required=True, choices=["CHAOS_MR_T2", "CHAOS_CT", "MSCMR", "MS-CMR", "SATA_CAP"])
    parser.add_argument("--image-dir", required=True)
    parser.add_argument("--label-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--image-glob", default="*.nii.gz")
    parser.add_argument("--label-glob", default="*.nii.gz")
    parser.add_argument("--p1", type=float, default=1.0)
    parser.add_argument("--p99", type=float, default=99.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = preprocess_dataset(
        dataset=args.dataset,
        image_dir=args.image_dir,
        label_dir=args.label_dir,
        output_dir=args.output_dir,
        image_glob=args.image_glob,
        label_glob=args.label_glob,
        p1=args.p1,
        p99=args.p99,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
