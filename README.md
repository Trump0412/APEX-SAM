# APEX-SAM: Training-Free Cross-Domain Few-Shot Medical Segmentation

<div align="center">

[![MICCAI 2026](https://img.shields.io/badge/MICCAI-2026-blue?style=flat-square)](https://miccai.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-informational?style=flat-square)](pyproject.toml)

</div>

APEX-SAM is a training-free framework for cross-domain few-shot medical image segmentation.

## Abstract

Training-free cross-domain few-shot medical image segmentation aims to segment unseen anatomies without parameter updates, addressing the high cost of dense annotation and domain-specific fine-tuning in clinical practice. Existing support-driven prompting methods face three limitations: support exemplars are randomly selected without quality assurance, geometric alignment is poorly modeled, and multi-modal prompt capabilities remain underexploited. We present APEX-SAM, a retrieval-augmented framework with three innovations. QAR builds a dual-stream DINO/SigLIP expert bank with diversity-aware selection to ensure support-query compatibility. APM performs style-aligned geometric matching and anatomy-guided point sampling from morphological priors. HMF fuses SAM branches (point, text, and box prompts in the full paper) via training-free feature-consensus weighting. Experiments on three cross-domain benchmarks confirm strong performance among training-free methods, with ablations validating each component's contribution.

## Method Overview

![APEX-SAM overview](assets/images/method/arch_3.png)

## Main Results (Paper)

### Abd-MRI and Abd-CT (Dice %)

| Method | Ref. | Abd-MRI Mean | Abd-CT Mean |
|---|---|---:|---:|
| PANet | ICCV'19 | 32.46 | 31.94 |
| SSL-ALP | TMI'22 | 63.01 | 47.46 |
| RPT | MICCAI'23 | 46.91 | 48.28 |
| PATNet | ECCV'22 | 52.97 | 57.29 |
| IFA | CVPR'24 | 40.61 | 30.79 |
| FAMNet | AAAI'25 | 65.79 | 64.75 |
| MAUP | MICCAI'25 | 67.09 | 67.46 |
| **APEX-SAM (Ours)** | — | **95.81** | **91.91** |

### Card-MRI (Dice %)

| Method | Ref. | LV-BP | LV-MYO | RV | Mean |
|---|---|---:|---:|---:|---:|
| PANet | ICCV'19 | 51.42 | 25.75 | 25.75 | 36.66 |
| SSL-ALP | TMI'22 | 83.47 | 22.73 | 66.21 | 57.47 |
| RPT | MICCAI'23 | 60.84 | 42.28 | 57.30 | 53.47 |
| PATNet | ECCV'22 | 65.35 | 50.63 | 68.34 | 61.44 |
| IFA | CVPR'24 | 50.43 | 31.32 | 30.74 | 37.50 |
| FAMNet | AAAI'25 | 86.64 | 51.82 | 76.26 | 71.58 |
| MAUP | MICCAI'25 | 88.36 | 52.74 | 78.29 | 73.13 |
| **APEX-SAM (Ours)** | — | **92.75** | **68.41** | **88.23** | **83.13** |

### Ablation (Dice %)

| Configuration | QAR | APM | HMF | Memory Rule | Mean Dice |
|---|---:|---:|---:|---|---:|
| Prompt-only baseline | ✗ | ✗ | ✗ | — | 72.4 |
| + QAR | ✓ | ✗ | ✗ | Fixed | 80.2 |
| + QAR + APM | ✓ | ✓ | ✗ | Fixed | 86.3 |
| + QAR + APM + HMF | ✓ | ✓ | ✓ | Fixed | 91.8 |
| + Full + thresholded append-only (ours) | ✓ | ✓ | ✓ | Thresholded append | **95.81** |

### Qualitative Results

![Qualitative and failure cases](assets/images/results/qual_failure.png)

## Open-Source Scope

To protect medical data privacy, this repository does not include the private expert database content.

What is released:

- **Module-2 (APM)**: open-source implementation.
- **Module-3 (HMF)**: open-source **vanilla bbox + point** implementation (same parameter setup as paper experiments).
- **Inference/Eval path**: accepts one externally selected support pair.

What is placeholder-only:

- **Module-1 (QAR expert database + DINOv3 rk2 retrieval)** is kept as file-level placeholders only.
- Placeholder files are intentionally empty:
  - `apex_sam/module1_qar/build_expert_database.py`
  - `apex_sam/module1_qar/retrieve_support_rank2.py`

You can build your own Module-1 with DINOv3 + SigLIP and place assets under `expert_database/`, then pass the selected support to this open-source inference pipeline.

## Repository Layout

```text
apex_sam/
  cli/
    build_expert_database.py   # Module-1 placeholder CLI
    preprocess_dataset.py      # dataset preprocessing
    inference.py               # single-case inference with one support
    eval.py                    # dataset evaluation with one selected support
  module1_qar/
    build_expert_database.py   # empty placeholder
    retrieve_support_rank2.py  # empty placeholder
  pipeline/
    segmenter.py               # APM + vanilla HMF core pipeline
  hmf/
    fusion.py                  # bbox+point vanilla fusion
  premask/
  prompting/
  sam/
expert_database/               # user-managed expert database assets
scripts/
  module1_qar_placeholder.sh
  run_single_inference.sh
  run_chaos_eval.sh
```

## Setup

```bash
git clone https://github.com/Trump0412/APEX-SAM.git
cd APEX-SAM

conda create -n apex-sam python=3.10 -y
conda activate apex-sam
pip install -e .
```

## Start

### 1) Download model repos

```bash
mkdir -p third_party
cd third_party

# SAM3
git clone https://github.com/facebookresearch/sam3.git

# DINOv3
git clone https://github.com/facebookresearch/dinov3.git

# SigLIP reference code
git clone https://github.com/google-research/big_vision.git
```

### 2) Download weights

```bash
# Hugging Face login (needed for gated models, e.g. SAM3)
hf auth login

# SAM3 / SAM3.1 (gated)
huggingface-cli download facebook/sam3 --local-dir ./checkpoints/sam3
huggingface-cli download facebook/sam3.1 --local-dir ./checkpoints/sam3_1

# DINOv3 ViT-L/16
huggingface-cli download facebook/dinov3-vitl16-pretrain-lvd1689m --local-dir ./checkpoints/dinov3_vitl16

# SigLIP SO400M
huggingface-cli download google/siglip-so400m-patch14-384 --local-dir ./checkpoints/siglip_so400m
```

Set paths for this repository:

```bash
export APEX_SAM_CHECKPOINT=/absolute/path/to/your_sam_checkpoint
export APEX_DINO_CHECKPOINT=/absolute/path/to/your_dinov3_checkpoint
export APEX_DINO_REPO=/absolute/path/to/third_party/dinov3
```

Use a checkpoint format that matches your installed SAM backend.

### 3) Prepare expert database directory and selected support

```text
expert_database/
  ... your externally built assets ...

support_item/
  image.npy
  mask_label1.npy
  mask_label2.npy
  ...
```

## Datasets

| Dataset | Usage | Link |
|---|---|---|
| CHAOS (CT/MR abdominal) | Abd-MRI / Abd-CT | https://chaos.grand-challenge.org/Data/ |
| MS-CMRSeg 2019 | Card-MRI | https://zmiclab.github.io/zxh/0/mscmrseg19/data.html |
| MICCAI 2013 SATA (CAP split) | additional cross-domain source | https://masi.vuse.vanderbilt.edu/submission/leaderboard.html |

## Preprocess Dataset

```bash
python -m apex_sam.cli.preprocess_dataset \
  --dataset CHAOS_MR_T2 \
  --image-dir /path/to/raw/images \
  --label-dir /path/to/raw/labels \
  --output-dir /path/to/CHAOS_MR_T2_preprocessed
```

Supported `--dataset` values:

- `CHAOS_MR_T2`
- `CHAOS_CT`
- `MSCMR` / `MS-CMR`
- `SATA_CAP`

## Inference

Single query with one selected support pair:

```bash
python -m apex_sam.cli.inference \
  --support-image-path /path/to/support_item/image.npy \
  --support-mask-path /path/to/support_item/mask_label1.npy \
  --query-image-path /path/to/query_slice.npy \
  --output-mask-path ./outputs/query_pred.npy \
  --sam-checkpoint $APEX_SAM_CHECKPOINT \
  --dinov3-checkpoint $APEX_DINO_CHECKPOINT \
  --dinov3-repo $APEX_DINO_REPO \
  --device cuda
```

## Eval

Evaluate on a normalized dataset using externally selected support:

```bash
python -m apex_sam.cli.eval \
  --dataset CHAOS_MR_T2 \
  --data-dir /path/to/CHAOS_MR_T2_preprocessed \
  --expert-database-dir /path/to/expert_database \
  --support-item-dir /path/to/support_item \
  --test-labels 1 2 3 4 \
  --max-cases 3 \
  --max-slices 8 \
  --sam-checkpoint $APEX_SAM_CHECKPOINT \
  --dinov3-checkpoint $APEX_DINO_CHECKPOINT \
  --dinov3-repo $APEX_DINO_REPO \
  --output-root ./outputs \
  --device cuda
```

Outputs are written under `./outputs/run_YYYYmmdd_HHMMSS/`.

## Citation

```bibtex
@inproceedings{apexsam2026,
  title     = {APEX-SAM: Anatomy-aware Prompting with Expert Retrieval for Training-free Medical Image Segmentation},
  author    = {Anonymous Authors},
  booktitle = {Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year      = {2026},
}
```
