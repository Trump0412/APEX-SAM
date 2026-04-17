from __future__ import annotations

import argparse

from apex_sam.config import ApexConfig
from apex_sam.constants import DEFAULT_OUTPUT_ROOT, default_dino_checkpoint, default_dino_repo, default_sam_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate APEX-SAM using externally selected support(s).")
    parser.add_argument("--dataset", default="CHAOS_MR_T2", choices=["CHAOS_MR_T2", "CHAOS_CT", "MSCMR", "MS-CMR", "SATA_CAP"])
    parser.add_argument("--data-dir", required=True)

    parser.add_argument(
        "--expert-database-dir",
        default="expert_database",
        help="Path to external expert database assets.",
    )
    parser.add_argument(
        "--support-item-dir",
        default="support_item",
        help="Directory containing one selected support: image.npy + mask_label{label}.npy",
    )
    parser.add_argument("--support-image-path", default="", help="Path to selected support image (.npy/.nii/.nii.gz)")
    parser.add_argument("--support-mask-path", default="", help="Path to selected support mask (.npy/.nii/.nii.gz)")
    parser.add_argument(
        "--support-mask-template",
        default="",
        help="Label-aware support mask template, e.g. /path/to/mask_label{label}.npy",
    )

    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-cases", type=int, default=-1, help="Use -1 for all cases.")
    parser.add_argument("--max-slices", type=int, default=-1, help="Use -1 for all valid slices.")
    parser.add_argument("--test-labels", type=int, nargs="*", default=[])
    parser.add_argument("--force-input-size", type=int, default=256)
    parser.add_argument(
        "--eval-protocol",
        choices=["slice_mean", "case_max_filtered"],
        default="case_max_filtered",
        help="Evaluation aggregation protocol.",
    )
    parser.add_argument(
        "--case-dice-threshold",
        type=float,
        default=0.1,
        help="Dice threshold used when eval-protocol=case_max_filtered.",
    )

    parser.add_argument("--enable-hmf", action="store_true", default=True)
    parser.add_argument("--disable-hmf", dest="enable_hmf", action="store_false")
    parser.add_argument("--hmf-temperature", type=float, default=1.0)
    parser.add_argument("--hmf-clip-eps", type=float, default=1e-4)

    parser.add_argument("--sam-checkpoint", default=default_sam_checkpoint())
    parser.add_argument("--dinov3-checkpoint", default=default_dino_checkpoint())
    parser.add_argument("--dinov3-repo", default=default_dino_repo())
    parser.add_argument("--device", default="cuda")
    return parser


def main() -> None:
    from apex_sam.evaluation.runner import run_evaluation

    args = build_parser().parse_args()
    config = ApexConfig.from_cli_args(args)
    summary = run_evaluation(config)
    print(summary.run_dir)


if __name__ == "__main__":
    main()
