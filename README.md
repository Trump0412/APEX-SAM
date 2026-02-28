# APEX-SAM

Minimal reproducibility release for the CHAOS-MRI setting.

## Status
- Minimal Repro Release
- CHAOS-MRI Only

## TODO
- [ ] Open-source minimal reproducibility and result-validation demo
- [ ] Release APM module
- [ ] Release QAR module
- [ ] Release HMF module

## Overview
APEX-SAM is the open-source minimal reproduction package for the CHAOS-MRI configuration used in our paper. This release keeps the local DINO database retrieval path, the query-native pre-mask generation path, the Voronoi prompt generation path, and SAM-based mask prediction. It removes unrelated datasets, remote retrieval backends, CT filtering logic, and dataset-specific tricks that are not part of the public minimal reproduction target.

## Released Scope
- CHAOS-MRI (`CHAOS_MR_T2`) only
- Local DINO descriptor database only
- Voronoi prompt mode only
- DINO frequency fusion enabled by default
- Fixed input resize to `256`
- Slice-level Dice reporting and saved prediction artifacts

## Installation
```bash
conda create -n apex-sam python=3.10 -y
conda activate apex-sam
pip install -e .
```

This repository depends on:
- `segment_anything`
- a local DINOv3 checkout or a `torch.hub` compatible repo

## Checkpoint Setup
Place checkpoints under `checkpoints/` or override them with CLI arguments or environment variables.

Supported environment variables:
- `APEX_SAM_CHECKPOINT`
- `APEX_DINO_CHECKPOINT`
- `APEX_DINO_REPO`

Expected files are documented in `checkpoints/README.md`.

## Dataset Layout
The release expects the CHAOS-MRI preprocessed dataset in the following layout:

```text
CHAOS_MR_T2_preprocessed/
└── normalized/
    ├── image_000.nii.gz
    ├── label_000.nii.gz
    ├── image_001.nii.gz
    ├── label_001.nii.gz
    └── ...
```

## Build Local DB
```bash
python -m apex_sam.cli.build_local_db \
  --data-dir /path/to/CHAOS_MR_T2_preprocessed \
  --local-db-path /path/to/CHAOS_MR_T2_local_dinov3_db.npz \
  --dinov3-checkpoint ./checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --dinov3-repo ./third_party/dinov3 \
  --device cuda
```

## Run Minimal Repro
```bash
python -m apex_sam.cli.eval \
  --data-dir /path/to/CHAOS_MR_T2_preprocessed \
  --local-db-path /path/to/CHAOS_MR_T2_local_dinov3_db.npz \
  --max-cases 3 \
  --max-slices 8 \
  --test-labels 1 \
  --retrieval-rank 2 \
  --output-root ./outputs \
  --sam-checkpoint ./checkpoints/sam_vit_h_4b8939.pth \
  --dinov3-checkpoint ./checkpoints/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth \
  --dinov3-repo ./third_party/dinov3 \
  --device cuda
```

## Output Structure
```text
outputs/run_YYYYMMDD_HHMMSS/
├── run.log
├── summary.json
├── metrics.csv
├── preds/
│   └── label1/
│       └── *_pred.npy
└── overlays/
    └── label1/
        └── *.png
```

## Repository Structure
```text
APEX-SAM/
├── apex_sam/
├── assets/
├── checkpoints/
├── configs/
├── scripts/
└── tests/
```

## Links
- Paper: `PAPER_URL_PLACEHOLDER`
- Code: `GITHUB_URL_PLACEHOLDER`
- HuggingFace: `HUGGINGFACE_URL_PLACEHOLDER`

## Abstract
Abstract placeholder.

## Citation
BibTeX placeholder.

## License
This project is released under the MIT License.
