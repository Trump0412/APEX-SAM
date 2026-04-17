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

"$PYTHON_BIN" -m apex_sam.cli.inference \
  --support-image-path "$1" \
  --support-mask-path "$2" \
  --query-image-path "$3" \
  --output-mask-path "$4" \
  --sam-checkpoint "$5" \
  --dinov3-checkpoint "$6" \
  --dinov3-repo "$7" \
  --device "${8:-cuda}"
