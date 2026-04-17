#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_BIN="python3"
  else
    echo "No python executable found. Set PYTHON_BIN=/path/to/python" >&2
    exit 127
  fi
fi

"$PYTHON_BIN" -m apex_sam.cli.eval \
  --dataset CHAOS_MR_T2 \
  --data-dir "$1" \
  --expert-database-dir "$2" \
  --support-item-dir "$3" \
  --max-cases "${4:-3}" \
  --max-slices "${5:-8}" \
  --test-labels ${6:-1} \
  --output-root "${7:-./outputs}" \
  --sam-checkpoint "$8" \
  --dinov3-checkpoint "$9" \
  --dinov3-repo "${10}" \
  --device "${11:-cuda}"
