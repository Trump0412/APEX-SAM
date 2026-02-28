#!/usr/bin/env bash
set -euo pipefail
python -m apex_sam.cli.build_local_db   --data-dir "$1"   --local-db-path "$2"   --dinov3-checkpoint "$3"   --dinov3-repo "$4"   --device "${5:-cuda}"
