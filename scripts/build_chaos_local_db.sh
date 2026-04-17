#!/usr/bin/env bash
set -euo pipefail

# Deprecated filename kept for backward compatibility.
# This now builds a public support pool instead of a private retrieval DB.
python -m apex_sam.cli.build_support_pool \
  --dataset CHAOS_MR_T2 \
  --data-dir "$1" \
  --output-dir "$2" \
  --labels 1 2 3 4 \
  --max-support-per-label "${3:-24}" \
  --min-mask-area "${4:-32}"
