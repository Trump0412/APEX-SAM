from __future__ import annotations

import argparse

from apex_sam.config import ApexConfig
from apex_sam.constants import DEFAULT_OUTPUT_ROOT, default_dino_checkpoint, default_dino_repo, default_sam_checkpoint


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the public APEX-SAM (APM + simplified HMF) evaluation pipeline.")
    parser.add_argument("--dataset", default="CHAOS_MR_T2", choices=["CHAOS_MR_T2", "CHAOS_CT", "MSCMR", "MS-CMR", "SATA_CAP"])
    parser.add_argument("--data-dir", required=True)
    parser.add_argument(
        "--support-pool-dir",
        default="",
        help="Path to user-provided support pool. Expected folders with image.npy + mask_label*.npy.",
    )
    parser.add_argument(
        "--local-db-path",
        default="",
        help="Legacy fallback only. Prefer --support-pool-dir in the open-source release.",
    )
    parser.add_argument("--output-root", default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--max-cases", type=int, default=3)
    parser.add_argument("--max-slices", type=int, default=8)
    parser.add_argument("--test-labels", type=int, nargs="*", default=[1])
    parser.add_argument("--retrieval-rank", type=int, default=2)
    parser.add_argument("--retrieval-topk", type=int, default=5)
    parser.add_argument("--retrieval-skip-self", action="store_true", default=True)
    parser.add_argument("--no-retrieval-skip-self", dest="retrieval_skip_self", action="store_false")
    parser.add_argument("--force-input-size", type=int, default=256)

    parser.add_argument("--enable-hmf", action="store_true", default=True)
    parser.add_argument("--disable-hmf", dest="enable_hmf", action="store_false")
    parser.add_argument("--hmf-temperature", type=float, default=0.15)
    parser.add_argument("--hmf-prior-bias", type=float, default=0.2)
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
