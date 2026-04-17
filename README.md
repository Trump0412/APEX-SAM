# APEX-SAM: Training-Free Cross-Domain Few-Shot Medical Segmentation

<div align="center">

[![MICCAI 2026](https://img.shields.io/badge/MICCAI-2026-blue?style=flat-square)](https://miccai.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green?style=flat-square)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10%2B-informational?style=flat-square)](pyproject.toml)

</div>

APEX-SAM is a **training-free** framework for cross-domain few-shot medical image segmentation.

## Abstract

Training-free cross-domain few-shot medical image segmentation aims to segment unseen anatomies without parameter updates, addressing the high cost of dense annotation and domain-specific fine-tuning in clinical practice. Existing support-driven prompting methods face three limitations: support exemplars are randomly selected without quality assurance, geometric alignment is poorly modeled, and multi-modal prompt capabilities remain underexploited. We present APEX-SAM, a retrieval-augmented framework with three innovations. QAR builds a dual-stream DINO/SigLIP expert bank with diversity-aware selection to ensure support-query compatibility. APM performs style-aligned geometric matching and anatomy-guided point sampling from morphological priors. HMF fuses three SAM branches (point, text, and box prompts) via training-free feature-consensus weighting. Experiments on three cross-domain benchmarks confirm state-of-the-art performance among training-free methods, with ablations validating each component's contribution.

## Method Overview

![APEX-SAM overview](assets/images/method/arch_3.png)

The full paper pipeline contains three components:

- `QAR`: Quality-Aware Retrieval
- `APM`: Anatomy-Aware Prompt Mining
- `HMF`: Hybrid Multi-Modal Fusion

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

## Important Open-Source Scope

To protect patient privacy, we **do not release the private QAR database construction pipeline** or its curated internal support bank.

This public repository releases:

- A reproducible **APM implementation**
- A reproducible **simplified HMF implementation** (point/box/prior training-free fusion)
- A public **support-pool interface** so users can provide their own support data
- Data preprocessing and support-pool construction scripts

This means:

- You can run end-to-end inference with your own support pool.
- You should **not** expect exact numbers from the full private paper system, because private QAR is not included.

## Repository Layout

```text
apex_sam/
  cli/
    eval.py                    # main inference/eval entry
    preprocess_dataset.py      # dataset normalization script
    build_support_pool.py      # build user-facing support pool
  data/
    normalized.py              # normalized dataset iterator
    io.py                      # I/O + label remap + resize
  pipeline/
    segmenter.py               # APM + simplified HMF core pipeline
  support/
    interface.py               # public support-provider protocol
    filesystem_pool.py         # folder-based support provider
    private_qar.py             # explicit private-stub placeholder
  hmf/
    fusion.py                  # reliability-weighted branch fusion
  premask/
  prompting/
  sam/
scripts/
  build_chaos_local_db.sh      # backward-compatible wrapper (now builds support pool)
  run_chaos_minimal.sh
```

## Installation

```bash
conda create -n apex-sam python=3.10 -y
conda activate apex-sam
pip install -e .
```

## Checkpoints

By default we read checkpoints from environment variables:

```bash
export APEX_SAM_CHECKPOINT=/path/to/sam_vit_h_4b8939.pth
export APEX_DINO_CHECKPOINT=/path/to/dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth
export APEX_DINO_REPO=/path/to/dinov3_repo
```

You can also pass explicit CLI arguments (`--sam-checkpoint`, `--dinov3-checkpoint`, `--dinov3-repo`).

## Datasets Used in This Project

| Dataset | Role in project | Access |
|---|---|---|
| CHAOS (CT/MR abdominal) | Abd-MRI / Abd-CT experiments | https://chaos.grand-challenge.org/Data/ |
| MS-CMRSeg 2019 (Cardiac MRI) | Card-MRI experiments | https://zmiclab.github.io/zxh/0/mscmrseg19/data.html |
| MICCAI 2013 SATA (CAP split) | Additional cross-domain support source in internal experiments | https://masi.vuse.vanderbilt.edu/submission/leaderboard.html |

Notes:

- `MS-CMRSeg` data are available after registration and DUA submission.
- `SATA CAP` is a legacy challenge dataset; access is organizer-controlled.

## Dataset Preprocessing

Convert your raw image/label NIfTI pairs to the standardized layout:

```bash
python -m apex_sam.cli.preprocess_dataset \
  --dataset CHAOS_MR_T2 \
  --image-dir /path/to/raw/images \
  --label-dir /path/to/raw/labels \
  --output-dir /path/to/CHAOS_MR_T2_preprocessed
```

Supported dataset names:

- `CHAOS_MR_T2`
- `CHAOS_CT`
- `MSCMR` / `MS-CMR`
- `SATA_CAP`

Expected output:

```text
<output_dir>/
  normalized/
    image_000.nii.gz
    label_000.nii.gz
    image_001.nii.gz
    label_001.nii.gz
    ...
  preprocess_manifest.json
```

## Build Public Support Pool (QAR Interface Replacement)

Users provide support examples through a folder-based pool.

Build from a normalized dataset:

```bash
python -m apex_sam.cli.build_support_pool \
  --dataset CHAOS_MR_T2 \
  --data-dir /path/to/CHAOS_MR_T2_preprocessed \
  --output-dir /path/to/support_pool \
  --labels 1 2 3 4 \
  --max-support-per-label 24
```

Generated format:

```text
support_pool/
  support_slices/
    case_000_slice_015/
      image.npy
      mask_label1.npy
      mask_label2.npy
      ...
      meta.json
  manifest/
    support_summary.csv
    summary.json
```

## Run Evaluation

```bash
python -m apex_sam.cli.eval \
  --dataset CHAOS_MR_T2 \
  --data-dir /path/to/CHAOS_MR_T2_preprocessed \
  --support-pool-dir /path/to/support_pool \
  --test-labels 1 2 3 4 \
  --max-cases 3 \
  --max-slices 8 \
  --retrieval-rank 2 \
  --retrieval-topk 5 \
  --sam-checkpoint $APEX_SAM_CHECKPOINT \
  --dinov3-checkpoint $APEX_DINO_CHECKPOINT \
  --dinov3-repo $APEX_DINO_REPO \
  --output-root ./outputs \
  --device cuda
```

Outputs:

- `run_YYYYmmdd_HHMMSS/metrics.csv`
- `run_YYYYmmdd_HHMMSS/summary.json`
- `run_YYYYmmdd_HHMMSS/preds/`
- `run_YYYYmmdd_HHMMSS/overlays/`

## Verified Server Reproduction (Executed)

The public pipeline was executed successfully on our remote server with existing preprocessed data and support slices.

Example completed run directories:

- `/root/autodl-tmp/ssb_output/apex_sam_open_release_runs/run_20260417_114904`
- `/root/autodl-tmp/ssb_output/apex_sam_open_release_runs/run_20260417_115544`

## Reproducibility Notes

- No hard-coded absolute data paths are required in code.
- QAR private components are intentionally excluded.
- The released HMF is simplified and training-free.

## Citation

```bibtex
@inproceedings{apexsam2026,
  title     = {APEX-SAM: Anatomy-aware Prompting with Expert Retrieval
               for Training-free Medical Image Segmentation},
  author    = {Anonymous Authors},
  booktitle = {Medical Image Computing and Computer Assisted Intervention (MICCAI)},
  year      = {2026},
}
```

## License

This project is released under the [MIT License](LICENSE).
