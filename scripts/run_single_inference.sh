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

SUPPORT_ITEM_DIR="$1"
QUERY_IMAGE_PATH="$2"
OUTPUT_MASK_PATH="${3:-./outputs/inference_pred.npy}"
EXTRA_ARGS=("${@:4}")

"$PYTHON_BIN" -m apex_sam.cli.inference \
  --support-item-dir "$SUPPORT_ITEM_DIR" \
  --query-image-path "$QUERY_IMAGE_PATH" \
  --output-mask-path "$OUTPUT_MASK_PATH" \
  "${EXTRA_ARGS[@]}"
