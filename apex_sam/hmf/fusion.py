from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BranchPrediction:
    name: str
    prob: np.ndarray
    confidence: float


class VanillaBBoxPointHMF:
    """Vanilla HMF implementation for public release: bbox + point branches only."""

    def __init__(self, temperature: float = 1.0, clip_eps: float = 1e-4):
        self.temperature = max(1e-3, float(temperature))
        self.clip_eps = float(np.clip(clip_eps, 1e-6, 1e-2))

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        values = values - values.max()
        exp = np.exp(values)
        return exp / (exp.sum() + 1e-8)

    def fuse(self, branches: list[BranchPrediction]) -> tuple[np.ndarray, dict]:
        if len(branches) != 2:
            raise RuntimeError("VanillaBBoxPointHMF expects exactly two branches: point and bbox.")

        names = [branch.name for branch in branches]
        probs = [np.asarray(branch.prob, dtype=np.float32) for branch in branches]
        confidences = np.asarray([float(branch.confidence) for branch in branches], dtype=np.float32)

        weights = self._softmax(confidences / self.temperature)

        fused_logit = np.zeros_like(probs[0], dtype=np.float32)
        for weight, prob in zip(weights, probs):
            clipped = np.clip(prob, self.clip_eps, 1.0 - self.clip_eps)
            fused_logit += float(weight) * np.log(clipped / (1.0 - clipped))

        fused_prob = 1.0 / (1.0 + np.exp(-fused_logit))
        fused_mask = (fused_prob > 0.5).astype(np.uint8)
        debug = {
            "weights": {name: float(weight) for name, weight in zip(names, weights)},
            "confidence": {name: float(score) for name, score in zip(names, confidences.tolist())},
        }
        return fused_mask, debug
