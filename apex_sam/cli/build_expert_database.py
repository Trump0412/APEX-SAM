from __future__ import annotations

"""Module-1 (QAR) placeholder CLI.

The private expert database construction and rank-2 support retrieval pipeline
are intentionally not included in this open-source release.
"""

import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Module-1 (QAR) placeholder. Implementation is intentionally omitted."
    )
    parser.add_argument(
        "--expert-database-dir",
        required=True,
        help="Reserved path for your own expert database assets.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    raise NotImplementedError(
        "Module-1 (QAR: expert database + rank-2 retrieval) is not released. "
        f"Use external matching and keep assets under: {args.expert_database_dir}"
    )


if __name__ == "__main__":
    main()
