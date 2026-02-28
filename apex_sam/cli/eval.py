from __future__ import annotations

import argparse

from apex_sam.config import ApexConfig
from apex_sam.constants import DEFAULT_OUTPUT_ROOT, default_dino_checkpoint, default_dino_repo, default_sam_checkpoint
from apex_sam.evaluation.runner import run_evaluation


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Run the APEX-SAM CHAOS-MRI minimal reproduction pipeline.')
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--local-db-path', required=True)
    parser.add_argument('--output-root', default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument('--max-cases', type=int, default=3)
    parser.add_argument('--max-slices', type=int, default=8)
    parser.add_argument('--test-labels', type=int, nargs='*', default=[1])
    parser.add_argument('--retrieval-rank', type=int, default=2)
    parser.add_argument('--sam-checkpoint', default=default_sam_checkpoint())
    parser.add_argument('--dinov3-checkpoint', default=default_dino_checkpoint())
    parser.add_argument('--dinov3-repo', default=default_dino_repo())
    parser.add_argument('--device', default='cuda')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    config = ApexConfig.from_cli_args(args)
    summary = run_evaluation(config)
    print(summary.run_dir)


if __name__ == '__main__':
    main()
