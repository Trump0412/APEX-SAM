#!/usr/bin/env bash
set -euo pipefail

python -m apex_sam.cli.eval \
  --dataset CHAOS_MR_T2 \
  --data-dir "$1" \
  --support-pool-dir "$2" \
  --max-cases "${3:-3}" \
  --max-slices "${4:-8}" \
  --test-labels "${5:-1}" \
  --retrieval-rank "${6:-2}" \
  --output-root "${7:-./outputs}" \
  --sam-checkpoint "$8" \
  --dinov3-checkpoint "$9" \
  --dinov3-repo "${10}" \
  --device "${11:-cuda}"
