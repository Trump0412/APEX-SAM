from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F


class SAMMaskPredictor:
    def __init__(self, checkpoint: str, device: str) -> None:
        self.checkpoint = checkpoint
        self.device = torch.device(device)
        self.predictor = self._load()

    def _load(self):
        print(f"[SAM] Loading model: {self.checkpoint}")
        try:
            from segment_anything import SamPredictor, sam_model_registry
            model = sam_model_registry['vit_h'](checkpoint=self.checkpoint)
            model.to(device=self.device)
            model.eval()
            predictor = SamPredictor(model)
            print('[SAM] Model loaded')
            return predictor
        except Exception as exc:
            print(f'[SAM] Load failed: {exc}')
            print('[SAM] Using stub predictor')
            return None

    def predict_with_points(
        self,
        image: np.ndarray,
        points_pos: np.ndarray,
        points_neg: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[List[np.ndarray], List[float], Optional[np.ndarray]]:
        if self.predictor is None:
            h, w = image.shape[:2]
            masks = [np.random.rand(h, w) > 0.5 for _ in range(3)]
            return masks, [0.5, 0.4, 0.3], None
        img_uint8 = (image * 255).astype(np.uint8)
        img_rgb = np.stack([img_uint8, img_uint8, img_uint8], axis=-1)
        self.predictor.set_image(img_rgb)
        point_coords = np.concatenate([points_pos, points_neg], axis=0).astype(np.float32)
        point_labels = np.array([1] * len(points_pos) + [0] * len(points_neg))
        h, w = image.shape[:2]
        point_coords[:, 0] = np.clip(point_coords[:, 0], 0, w - 1)
        point_coords[:, 1] = np.clip(point_coords[:, 1], 0, h - 1)
        box = None
        if bbox is not None:
            x0, y0, x1, y1 = bbox
            x0 = int(np.clip(x0, 0, w - 1))
            y0 = int(np.clip(y0, 0, h - 1))
            x1 = int(np.clip(x1, 0, w - 1))
            y1 = int(np.clip(y1, 0, h - 1))
            if x1 <= x0:
                x1 = min(w - 1, x0 + 1)
            if y1 <= y0:
                y1 = min(h - 1, y0 + 1)
            box = np.array([x0, y0, x1, y1], dtype=np.float32)
        masks, scores, _ = self.predictor.predict(point_coords=point_coords, point_labels=point_labels, box=box, multimask_output=False)
        masks_list = [masks[i] for i in range(len(masks))]
        scores_list = scores.tolist()
        embedding = None
        try:
            feat = self.predictor.get_image_embedding()
            if isinstance(feat, torch.Tensor):
                feat_up = F.interpolate(feat, size=img_rgb.shape[:2], mode='bilinear', align_corners=False)
                embedding = feat_up.permute(0, 2, 3, 1).squeeze(0).contiguous().cpu().numpy().astype(np.float32)
        except Exception:
            embedding = None
        return masks_list, scores_list, embedding


class SamMixin:
    def _load_sam2(self):
        self.sam_backend = SAMMaskPredictor(self.config.sam2_checkpoint, self.config.device)
        return self.sam_backend.predictor

    def _run_sam2(self, Iq: np.ndarray, P_pos: np.ndarray, P_neg: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None):
        return self.sam_backend.predict_with_points(Iq, P_pos, P_neg, bbox=bbox)
