from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class BranchPrediction:
    name: str
    prob: np.ndarray
    confidence: float


class HMFSimpleFusion:
    """
    Training-free reliability-weighted fusion used in the open-source release.

    Branches:
      - point branch
      - box branch
      - anatomy prior branch (from APM pre-mask)
    """

    def __init__(self, temperature: float = 0.15, clip_eps: float = 1e-4, prior_bias: float = 0.2):
        self.temperature = max(1e-3, float(temperature))
        self.clip_eps = float(np.clip(clip_eps, 1e-6, 1e-2))
        self.prior_bias = float(prior_bias)

    @staticmethod
    def _dice(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
        a = (mask_a > 0.5).astype(np.float32)
        b = (mask_b > 0.5).astype(np.float32)
        inter = float((a * b).sum())
        denom = float(a.sum() + b.sum())
        if denom <= 0:
            return 1.0
        return float(2.0 * inter / (denom + 1e-8))

    @staticmethod
    def _softmax(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float32)
        values = values - values.max()
        exp = np.exp(values)
        return exp / (exp.sum() + 1e-8)

    def fuse(self, branches: list[BranchPrediction]) -> tuple[np.ndarray, dict]:
        if not branches:
            raise RuntimeError("HMF fusion requires at least one branch.")
        if len(branches) == 1:
            only = branches[0].prob.astype(np.float32)
            return (only > 0.5).astype(np.uint8), {
                "weights": {branches[0].name: 1.0},
                "consensus": {branches[0].name: 1.0},
                "confidence": {branches[0].name: float(branches[0].confidence)},
            }

        consensus_scores: list[float] = []
        confidence_scores: list[float] = []
        names: list[str] = []
        probs: list[np.ndarray] = []

        for idx, branch in enumerate(branches):
            names.append(branch.name)
            probs.append(np.asarray(branch.prob, dtype=np.float32))
            confidence_scores.append(float(branch.confidence))
            pairwise = []
            for jdx, other in enumerate(branches):
                if idx == jdx:
                    continue
                pairwise.append(self._dice(branch.prob, other.prob))
            consensus = float(np.mean(pairwise)) if pairwise else 1.0
            if branch.name == "prior":
                consensus += self.prior_bias
            consensus_scores.append(consensus)

        logits = np.asarray(consensus_scores, dtype=np.float32) + np.asarray(confidence_scores, dtype=np.float32)
        weights = self._softmax(logits / self.temperature)

        fused_logit = np.zeros_like(probs[0], dtype=np.float32)
        for weight, prob in zip(weights, probs):
            clipped = np.clip(prob, self.clip_eps, 1.0 - self.clip_eps)
            fused_logit += float(weight) * np.log(clipped / (1.0 - clipped))
        fused_prob = 1.0 / (1.0 + np.exp(-fused_logit))
        fused_mask = (fused_prob > 0.5).astype(np.uint8)

        debug = {
            "weights": {name: float(weight) for name, weight in zip(names, weights)},
            "consensus": {name: float(score) for name, score in zip(names, consensus_scores)},
            "confidence": {name: float(score) for name, score in zip(names, confidence_scores)},
        }
        return fused_mask, debug
