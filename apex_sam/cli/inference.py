from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from apex_sam.config import ApexConfig
from apex_sam.constants import default_dino_checkpoint, default_dino_repo, default_sam_checkpoint
from apex_sam.data.io import load_nifti, resize_image_2d, resize_mask_2d


def _load_array(path: str) -> np.ndarray:
    lower = path.lower()
    if lower.endswith(".npy"):
        return np.load(path)
    if lower.endswith(".nii") or lower.endswith(".nii.gz"):
        return load_nifti(path)
    raise RuntimeError(f"Unsupported file format: {path}")


def _select_slice(arr: np.ndarray, slice_index: int | None) -> np.ndarray:
    arr = np.asarray(arr)
    if arr.ndim == 2:
        return arr
    if arr.ndim == 3:
        if slice_index is None:
            slice_index = arr.shape[0] // 2
        slice_index = int(np.clip(slice_index, 0, arr.shape[0] - 1))
        return arr[slice_index]
    raise RuntimeError(f"Expected 2D/3D array, got shape={arr.shape}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-case inference with one externally selected support pair.")
    parser.add_argument("--support-image-path", required=True)
    parser.add_argument("--support-mask-path", required=True)
    parser.add_argument("--query-image-path", required=True)
    parser.add_argument("--output-mask-path", required=True)
    parser.add_argument("--support-slice-index", type=int, default=None)
    parser.add_argument("--query-slice-index", type=int, default=None)
    parser.add_argument("--force-input-size", type=int, default=256)

    parser.add_argument("--sam-checkpoint", default=default_sam_checkpoint())
    parser.add_argument("--dinov3-checkpoint", default=default_dino_checkpoint())
    parser.add_argument("--dinov3-repo", default=default_dino_repo())
    parser.add_argument("--device", default="cuda")

    parser.add_argument("--enable-hmf", action="store_true", default=True)
    parser.add_argument("--disable-hmf", dest="enable_hmf", action="store_false")
    parser.add_argument("--hmf-temperature", type=float, default=1.0)
    parser.add_argument("--hmf-clip-eps", type=float, default=1e-4)
    return parser


def main() -> None:
    from apex_sam.pipeline.segmenter import ApexSegmenter

    args = build_parser().parse_args()
    config = ApexConfig(
        force_input_size=int(args.force_input_size),
        sam_checkpoint=args.sam_checkpoint,
        dinov3_checkpoint=args.dinov3_checkpoint,
        dinov3_repo=args.dinov3_repo,
        device=args.device,
        enable_hmf=bool(args.enable_hmf),
        hmf_temperature=float(args.hmf_temperature),
        hmf_clip_eps=float(args.hmf_clip_eps),
    )

    support_image = _select_slice(_load_array(args.support_image_path), args.support_slice_index).astype(np.float32)
    support_mask = (_select_slice(_load_array(args.support_mask_path), args.support_slice_index) > 0.5).astype(np.float32)
    query_image = _select_slice(_load_array(args.query_image_path), args.query_slice_index).astype(np.float32)

    support_image_in = resize_image_2d(support_image, (config.force_input_size, config.force_input_size))
    support_mask_in = resize_mask_2d(support_mask, (config.force_input_size, config.force_input_size))
    query_image_in = resize_image_2d(query_image, (config.force_input_size, config.force_input_size))

    segmenter = ApexSegmenter(config)
    result = segmenter.predict(
        support_image_in,
        support_mask_in,
        query_image_in,
        case_id="inference",
        slice_id=0,
        gt_mask=None,
        viz_dir=None,
        logger=None,
    )

    pred_mask = resize_mask_2d(result.pred_mask.astype(np.float32), query_image.shape)
    output_path = Path(args.output_mask_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, pred_mask.astype(np.uint8))
    print(str(output_path))


if __name__ == "__main__":
    main()
