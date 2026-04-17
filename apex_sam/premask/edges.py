from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage import binary_erosion
from skimage import measure


class EdgeMixin:
    def _compute_edge_map(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute query edge candidates using multi-scale LoG + Sobel gradients."""
        # Ensure float32 for OpenCV ops
        img = img.astype(np.float32, copy=False)

        # Multi-scale LoG
        E_log = np.zeros_like(img)
        for sigma in self.config.sigmas:
            log_response = ndimage.gaussian_laplace(img, sigma=sigma)
            E_log = np.maximum(E_log, np.abs(log_response))

        # Gradient magnitude (Sobel)
        sobel_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        E_grad = np.sqrt(sobel_x**2 + sobel_y**2)

        # Fuse and normalize
        E_q = self.config.w_log * E_log + self.config.w_grad * E_grad
        E_q = self._normalize(E_q)
        E_grad = self._normalize(E_grad)

        return E_q, E_grad

    def _compute_structure_maps(self, img: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute edge strength + orientation + valid mask for oriented matching.
        """
        img = img.astype(np.float32, copy=False)
        E_q, _ = self._compute_edge_map(img)
        sobel_x = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
        sobel_y = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
        theta = np.arctan2(sobel_y, sobel_x)
        valid = E_q >= float(self.config.edge_valid_thresh)
        return E_q, theta, valid

    def _support_boundary_template(self, mask: np.ndarray) -> Optional[Dict[str, Any]]:
        """Extract ordered boundary points + normals from support mask."""
        mask_bin = (mask > 0.5).astype(np.uint8)
        contours = measure.find_contours(mask_bin, 0.5)
        if not contours:
            return None
        contour = max(contours, key=len)
        contour = np.asarray(contour, dtype=np.float32)
        if contour.shape[0] < 3:
            return None
        # Tangent/normal on full contour (y, x)
        prev = np.roll(contour, 1, axis=0)
        nxt = np.roll(contour, -1, axis=0)
        tangent = nxt - prev
        theta_t = np.arctan2(tangent[:, 0], tangent[:, 1])
        theta_n = theta_t + np.pi / 2.0
        theta_n = np.arctan2(np.sin(theta_n), np.cos(theta_n))

        # Sample points by arc length
        diffs = np.diff(contour, axis=0, append=contour[:1])
        seglen = np.linalg.norm(diffs, axis=1)
        total = float(seglen.sum())
        if total <= 1e-6:
            return None
        step = max(1.0, float(self.config.boundary_sample_step))
        sample_d = np.arange(0.0, total, step)
        cum = np.cumsum(seglen)
        idx = np.searchsorted(cum, sample_d)
        idx = np.clip(idx, 0, contour.shape[0] - 1)
        pts = contour[idx]
        th = theta_n[idx]
        pts_xy = pts[:, ::-1].astype(np.float32)
        center_xy = pts_xy.mean(axis=0).astype(np.float32)
        return {
            "points": pts_xy,
            "theta": th.astype(np.float32),
            "contour": contour,
            "center": center_xy,
        }
