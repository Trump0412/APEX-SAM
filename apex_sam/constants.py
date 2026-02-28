from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_ROOT = str(REPO_ROOT / "outputs")
DEFAULT_SAM_CHECKPOINT_NAME = "sam_vit_h_4b8939.pth"
DEFAULT_DINO_CHECKPOINT_NAME = "dinov3_vitl16_pretrain_lvd1689m-8aa4cbdd.pth"
DEFAULT_DINO_MODEL_NAME = "dinov3_vitl16"


def default_sam_checkpoint() -> str:
    return os.getenv("APEX_SAM_CHECKPOINT", str(REPO_ROOT / "checkpoints" / DEFAULT_SAM_CHECKPOINT_NAME))


def default_dino_checkpoint() -> str:
    return os.getenv("APEX_DINO_CHECKPOINT", str(REPO_ROOT / "checkpoints" / DEFAULT_DINO_CHECKPOINT_NAME))


def default_dino_repo() -> str:
    return os.getenv("APEX_DINO_REPO", str(REPO_ROOT / "third_party" / "dinov3"))
