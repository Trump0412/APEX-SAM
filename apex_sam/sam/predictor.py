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

            model = sam_model_registry["vit_h"](checkpoint=self.checkpoint)
            model.to(device=self.device)
            model.eval()
            predictor = SamPredictor(model)
            print("[SAM] Model loaded")
            return predictor
        except Exception as exc:
            print(f"[SAM] Load failed: {exc}")
            print("[SAM] Using stub predictor")
            return None

    @staticmethod
    def _sanitize_box(bbox: Tuple[int, int, int, int], h: int, w: int) -> np.ndarray:
        x0, y0, x1, y1 = bbox
        x0 = int(np.clip(x0, 0, w - 1))
        y0 = int(np.clip(y0, 0, h - 1))
        x1 = int(np.clip(x1, 0, w - 1))
        y1 = int(np.clip(y1, 0, h - 1))
        if x1 <= x0:
            x1 = min(w - 1, x0 + 1)
        if y1 <= y0:
            y1 = min(h - 1, y0 + 1)
        return np.array([x0, y0, x1, y1], dtype=np.float32)

    def _predict(
        self,
        image: np.ndarray,
        points_pos: Optional[np.ndarray] = None,
        points_neg: Optional[np.ndarray] = None,
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[List[np.ndarray], List[float], Optional[np.ndarray]]:
        h, w = image.shape[:2]
        if self.predictor is None:
            random_mask = (np.random.rand(h, w) > 0.5).astype(np.float32)
            return [random_mask], [0.5], None

        img_uint8 = (np.clip(image, 0.0, 1.0) * 255).astype(np.uint8)
        img_rgb = np.stack([img_uint8, img_uint8, img_uint8], axis=-1)
        self.predictor.set_image(img_rgb)

        point_coords = None
        point_labels = None
        pos = np.asarray(points_pos, dtype=np.float32) if points_pos is not None else np.zeros((0, 2), dtype=np.float32)
        neg = np.asarray(points_neg, dtype=np.float32) if points_neg is not None else np.zeros((0, 2), dtype=np.float32)
        if len(pos) > 0 or len(neg) > 0:
            point_coords = np.concatenate([pos, neg], axis=0).astype(np.float32)
            point_labels = np.array([1] * len(pos) + [0] * len(neg), dtype=np.int32)
            point_coords[:, 0] = np.clip(point_coords[:, 0], 0, w - 1)
            point_coords[:, 1] = np.clip(point_coords[:, 1], 0, h - 1)

        box = self._sanitize_box(bbox, h, w) if bbox is not None else None

        masks, scores, _ = self.predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            multimask_output=False,
        )
        masks_list = [masks[i].astype(np.float32) for i in range(len(masks))]
        scores_list = [float(v) for v in scores.tolist()]

        embedding = None
        try:
            feat = self.predictor.get_image_embedding()
            if isinstance(feat, torch.Tensor):
                feat_up = F.interpolate(feat, size=img_rgb.shape[:2], mode="bilinear", align_corners=False)
                embedding = feat_up.permute(0, 2, 3, 1).squeeze(0).contiguous().cpu().numpy().astype(np.float32)
        except Exception:
            embedding = None
        return masks_list, scores_list, embedding

    def predict_with_points(
        self,
        image: np.ndarray,
        points_pos: np.ndarray,
        points_neg: np.ndarray,
        bbox: Optional[Tuple[int, int, int, int]] = None,
    ) -> Tuple[List[np.ndarray], List[float], Optional[np.ndarray]]:
        return self._predict(image=image, points_pos=points_pos, points_neg=points_neg, bbox=bbox)

    def predict_with_box(
        self,
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Tuple[List[np.ndarray], List[float], Optional[np.ndarray]]:
        return self._predict(image=image, points_pos=None, points_neg=None, bbox=bbox)

    def predict_with_points_and_box(
        self,
        image: np.ndarray,
        points_pos: np.ndarray,
        points_neg: np.ndarray,
        bbox: Tuple[int, int, int, int],
    ) -> Tuple[List[np.ndarray], List[float], Optional[np.ndarray]]:
        return self._predict(image=image, points_pos=points_pos, points_neg=points_neg, bbox=bbox)


class SamMixin:
    def _load_sam2(self):
        self.sam_backend = SAMMaskPredictor(self.config.sam2_checkpoint, self.config.device)
        return self.sam_backend.predictor

    def _run_sam2(self, Iq: np.ndarray, P_pos: np.ndarray, P_neg: np.ndarray, bbox: Optional[Tuple[int, int, int, int]] = None):
        return self.sam_backend.predict_with_points(Iq, P_pos, P_neg, bbox=bbox)

    def _run_sam_box(self, Iq: np.ndarray, bbox: Tuple[int, int, int, int]):
        return self.sam_backend.predict_with_box(Iq, bbox)

    def _run_sam_points_and_box(self, Iq: np.ndarray, P_pos: np.ndarray, P_neg: np.ndarray, bbox: Tuple[int, int, int, int]):
        return self.sam_backend.predict_with_points_and_box(Iq, P_pos, P_neg, bbox)
