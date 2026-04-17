from __future__ import annotations

"""
Backward-compatible wrapper.

The private QAR DB builder is not part of the open-source release.
This command now builds a public support pool instead.
"""

import argparse

from apex_sam.cli.build_support_pool import build_support_pool


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="[Deprecated] Build support resources. This now exports a public support pool instead of private QAR DB."
    )
    parser.add_argument("--dataset", default="CHAOS_MR_T2", choices=["CHAOS_MR_T2", "CHAOS_CT", "MSCMR", "MS-CMR", "SATA_CAP"])
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--local-db-path", required=True, help="Deprecated name. Interpreted as support-pool output directory.")
    parser.add_argument("--labels", type=int, nargs="*", default=[1, 2, 3, 4])
    parser.add_argument("--max-support-per-label", type=int, default=24)
    parser.add_argument("--min-mask-area", type=int, default=32)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    summary = build_support_pool(
        data_dir=args.data_dir,
        output_dir=args.local_db_path,
        dataset=args.dataset,
        labels=list(args.labels),
        max_support_per_label=args.max_support_per_label,
        min_mask_area=args.min_mask_area,
    )
    print(summary["support_pool_dir"])


if __name__ == "__main__":
    main()
