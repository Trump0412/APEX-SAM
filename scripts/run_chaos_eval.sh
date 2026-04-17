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

DATA_DIR="$1"
EXPERT_DATABASE_DIR="${2:-expert_database}"
SUPPORT_ITEM_DIR="${3:-support_item}"
OUTPUT_ROOT="${4:-./outputs}"
EXTRA_ARGS=("${@:5}")

"$PYTHON_BIN" -m apex_sam.cli.eval \
  --data-dir "$DATA_DIR" \
  --expert-database-dir "$EXPERT_DATABASE_DIR" \
  --support-item-dir "$SUPPORT_ITEM_DIR" \
  --output-root "$OUTPUT_ROOT" \
  "${EXTRA_ARGS[@]}"
