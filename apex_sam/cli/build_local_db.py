from __future__ import annotations

import argparse

from apex_sam.constants import default_dino_checkpoint, default_dino_repo
from apex_sam.retrieval.dino_encoder import DINOEncoder
from apex_sam.retrieval.local_db import LocalSupportDB


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description='Build a local DINO descriptor DB for CHAOS-MRI.')
    parser.add_argument('--data-dir', required=True)
    parser.add_argument('--local-db-path', required=True)
    parser.add_argument('--dinov3-checkpoint', default=default_dino_checkpoint())
    parser.add_argument('--dinov3-repo', default=default_dino_repo())
    parser.add_argument('--device', default='cuda')
    return parser


def main() -> None:
    args = build_parser().parse_args()
    encoder = DINOEncoder(
        checkpoint=args.dinov3_checkpoint,
        repo=args.dinov3_repo,
        model_name='dinov3_vitl16',
        device=args.device,
    )
    out = LocalSupportDB.build(args.data_dir, args.local_db_path, encoder)
    print(out)


if __name__ == '__main__':
    main()
