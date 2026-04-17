from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from sklearn.cluster import KMeans


class VoronoiPromptMixin:
    def _sample_points_legacy(self, Pin: np.ndarray, Pband: np.ndarray, Sdino: np.ndarray,
                              Dalign: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """A7. Legacy positive/negative point sampling."""
        H, W = Pin.shape

        # === Positive points ===
        Score_pos = Pin * Sdino
        mask_pos = Pin > 0.6
        Score_pos[~mask_pos] = 0

        # Top M_pos_cand pixels
        flat_scores = Score_pos.flatten()
        top_indices = np.argsort(flat_scores)[-self.config.M_pos_cand:]
        top_indices = top_indices[flat_scores[top_indices] > 0]  # filter zeros

        if len(top_indices) < self.config.K_pos:
            # Not enough candidates, sample from all positive scores
            top_indices = np.where(flat_scores > 0)[0]

        if len(top_indices) == 0:
            # Fallback: random points
            P_pos = np.random.randint(0, [W, H], size=(self.config.K_pos, 2))
        else:
            # Convert to (y, x)
            coords = np.array(np.unravel_index(top_indices, (H, W))).T  # (N, 2) - (y, x)

            # KMeans clustering
            n_clusters = min(self.config.K_pos, len(coords))
            if n_clusters > 1:
                kmeans = KMeans(n_clusters=n_clusters, max_iter=self.config.kmeans_max_iter, 
                               random_state=self.config.seed, n_init=10)
                labels = kmeans.fit_predict(coords)

                # Pick max score per cluster
                P_pos = []
                for i in range(n_clusters):
                    cluster_mask = labels == i
                    cluster_indices = top_indices[cluster_mask]
                    cluster_scores = flat_scores[cluster_indices]
                    best_idx = cluster_indices[np.argmax(cluster_scores)]
                    y, x = np.unravel_index(best_idx, (H, W))
                    P_pos.append([x, y])
                P_pos = np.array(P_pos)
            else:
                # Single point
                best_idx = top_indices[np.argmax(flat_scores[top_indices])]
                y, x = np.unravel_index(best_idx, (H, W))
                P_pos = np.array([[x, y]])

        # === Negative points (tight to boundary) ===
        bw = max(1, int(self.config.neg_boundary_width))
        boundary_band = (np.abs(Dalign) <= bw) & (Dalign <= 0)
        Score_neg = boundary_band.astype(np.float32) * (1 - Sdino)

        mask_fg = Pin > 0.5
        if mask_fg.any():
            ys_fg, xs_fg = np.nonzero(mask_fg)
            cx = float(xs_fg.mean())
            cy = float(ys_fg.mean())
        else:
            cx = W / 2.0
            cy = H / 2.0

        ys_c, xs_c = np.nonzero(boundary_band)
        if len(xs_c) == 0:
            P_neg = np.random.randint(0, [W, H], size=(self.config.K_neg, 2))
            return P_pos, P_neg

        scores_c = Score_neg[ys_c, xs_c]
        sector_size = 2.0 * np.pi / 8.0
        angles = np.arctan2(ys_c - cy, xs_c - cx)
        sectors = np.floor((angles + np.pi) / sector_size).astype(int)
        sectors = np.clip(sectors, 0, 7)

        selected_points = []
        used = set()
        # Ensure 8-direction surround first.
        for sec in range(8):
            idxs = np.where(sectors == sec)[0]
            if len(idxs) == 0:
                continue
            best_local = idxs[int(np.argmax(scores_c[idxs]))]
            x = int(xs_c[best_local])
            y = int(ys_c[best_local])
            key = (x, y)
            if key in used:
                continue
            used.add(key)
            selected_points.append([x, y])
            if len(selected_points) >= self.config.K_neg:
                break

        if len(selected_points) < self.config.K_neg:
            order = np.argsort(scores_c)[::-1]
            for j in order:
                x = int(xs_c[j])
                y = int(ys_c[j])
                key = (x, y)
                if key in used:
                    continue
                used.add(key)
                selected_points.append([x, y])
                if len(selected_points) >= self.config.K_neg:
                    break
        if len(selected_points) == 0:
            P_neg = np.random.randint(0, [W, H], size=(self.config.K_neg, 2))
        else:
            P_neg = np.array(selected_points[:self.config.K_neg])

        return P_pos, P_neg

    def _sample_points(self, Pin: np.ndarray, Pband: np.ndarray, Sdino: np.ndarray,
                       Dalign: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Backward-compatible entry point using legacy sampling."""
        return self._sample_points_legacy(Pin, Pband, Sdino, Dalign)

    def _sample_negative_points(self, Pband: np.ndarray, Sdino: np.ndarray, Dpre: np.ndarray,
                                Pin: Optional[np.ndarray] = None) -> np.ndarray:
        """Sample negative points close to the pre-mask boundary."""
        H, W = Pband.shape
        bw = max(1, int(self.config.neg_boundary_width))
        boundary_band = (np.abs(Dpre) <= bw) & (Dpre <= 0)
        Score_neg = boundary_band.astype(np.float32) * (1 - Sdino)

        if Pin is None:
            mask_fg = Dpre > 0
        else:
            mask_fg = Pin > 0.5
        if mask_fg.any():
            ys_fg, xs_fg = np.nonzero(mask_fg)
            cx = float(xs_fg.mean())
            cy = float(ys_fg.mean())
        else:
            cx = W / 2.0
            cy = H / 2.0

        ys_c, xs_c = np.nonzero(boundary_band)
        if len(xs_c) == 0:
            return np.random.randint(0, [W, H], size=(self.config.K_neg, 2))

        scores_c = Score_neg[ys_c, xs_c]
        sector_size = 2.0 * np.pi / 8.0
        angles = np.arctan2(ys_c - cy, xs_c - cx)
        sectors = np.floor((angles + np.pi) / sector_size).astype(int)
        sectors = np.clip(sectors, 0, 7)

        selected_points = []
        used = set()
        for sec in range(8):
            idxs = np.where(sectors == sec)[0]
            if len(idxs) == 0:
                continue
            best_local = idxs[int(np.argmax(scores_c[idxs]))]
            x = int(xs_c[best_local])
            y = int(ys_c[best_local])
            key = (x, y)
            if key in used:
                continue
            used.add(key)
            selected_points.append([x, y])
            if len(selected_points) >= self.config.K_neg:
                break

        if len(selected_points) < self.config.K_neg:
            order = np.argsort(scores_c)[::-1]
            for j in order:
                x = int(xs_c[j])
                y = int(ys_c[j])
                key = (x, y)
                if key in used:
                    continue
                used.add(key)
                selected_points.append([x, y])
                if len(selected_points) >= self.config.K_neg:
                    break

        if len(selected_points) == 0:
            return np.random.randint(0, [W, H], size=(self.config.K_neg, 2))
        return np.array(selected_points[:self.config.K_neg])

    def _sample_points_voronoi(
        self,
        M_pre: np.ndarray,
        Pin_pre: np.ndarray,
        Pband_pre: np.ndarray,
        Sdino: np.ndarray,
        Dpre: np.ndarray,
        Iq: Optional[np.ndarray] = None,
        case_id: Optional[str] = None,
        slice_id: Optional[int] = None,
        viz_dir: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Voronoi/FPS positive sampling with legacy negative sampling."""
        mask = (M_pre > 0.5)
        H, W = mask.shape
        if mask.sum() == 0:
            return self._sample_points_legacy(Pin_pre, Pband_pre, Sdino, Dpre)

        D_in = distance_transform_edt(mask)
        if self.config.use_distance_transform:
            interior = mask & (D_in >= self.config.d0)
        else:
            interior = mask.copy()
        if interior.sum() == 0:
            interior = mask.copy()

        coords_yx = np.stack(np.nonzero(interior), axis=1)
        if self.config.candidate_downsample > 1 and len(coords_yx) > 0:
            coords_yx = coords_yx[:: int(self.config.candidate_downsample)]
        if len(coords_yx) == 0:
            return self._sample_points_legacy(Pin_pre, Pband_pre, Sdino, Dpre)

        # FPS seed selection
        y0, x0 = np.unravel_index(np.argmax(D_in * interior), D_in.shape)
        P_sites = [(int(x0), int(y0))]
        coords_xy = coords_yx[:, ::-1].astype(np.float32)
        dist_min = np.full(len(coords_xy), np.inf, dtype=np.float32)
        for _ in range(1, self.config.K_pos):
            last = np.array(P_sites[-1], dtype=np.float32)
            d = np.linalg.norm(coords_xy - last[None, :], axis=1)
            dist_min = np.minimum(dist_min, d)
            max_idx = int(np.argmax(dist_min))
            if dist_min[max_idx] < self.config.min_dist:
                break
            px, py = coords_xy[max_idx]
            P_sites.append((int(px), int(py)))

        # Voronoi partition
        P_pos = []
        label_map = None
        if self.config.enable_voronoi_partition:
            sites = np.array(P_sites, dtype=np.float32)
            coords_all_yx = np.stack(np.nonzero(interior), axis=1)
            coords_all_xy = coords_all_yx[:, ::-1].astype(np.float32)
            if self.config.voronoi_assign_method == "kd_tree":
                tree = cKDTree(sites)
                _, idxs = tree.query(coords_all_xy, k=1)
            else:
                dists = np.linalg.norm(coords_all_xy[:, None, :] - sites[None, :, :], axis=2)
                idxs = np.argmin(dists, axis=1)
            label_map = -np.ones((H, W), dtype=np.int32)
            label_map[coords_all_yx[:, 0], coords_all_yx[:, 1]] = idxs
            for i in range(len(P_sites)):
                ys, xs = np.where(label_map == i)
                if len(xs) == 0:
                    continue
                best_idx = int(np.argmax(D_in[ys, xs]))
                P_pos.append([int(xs[best_idx]), int(ys[best_idx])])

            if self.config.save_voronoi_viz and viz_dir is not None:
                self._save_voronoi_viz(
                    Iq=Iq,
                    M_pre=mask.astype(np.uint8),
                    label_map=label_map,
                    P_sites=P_sites,
                    P_pos=P_pos,
                    case_id=case_id,
                    slice_id=slice_id,
                    viz_dir=viz_dir,
                )

        if len(P_pos) < self.config.K_pos:
            used = set(tuple(p) for p in P_pos)
            for p in P_sites:
                if p in used:
                    continue
                P_pos.append([int(p[0]), int(p[1])])
                if len(P_pos) >= self.config.K_pos:
                    break

        if len(P_pos) == 0:
            P_pos = self._sample_points_legacy(Pin_pre, Pband_pre, Sdino, Dpre)[0]
        else:
            P_pos = np.array(P_pos[: self.config.K_pos], dtype=np.int32)

        P_neg = self._sample_negative_points(Pband_pre, Sdino, Dpre, Pin_pre)
        return P_pos, P_neg

    def _dedup_points(self, points: np.ndarray, max_points: Optional[int] = None) -> np.ndarray:
        if points is None or len(points) == 0:
            return np.zeros((0, 2), dtype=np.int32)
        pts = np.round(points).astype(np.int32)
        seen = set()
        uniq = []
        for p in pts:
            key = (int(p[0]), int(p[1]))
            if key in seen:
                continue
            seen.add(key)
            uniq.append([key[0], key[1]])
            if max_points is not None and len(uniq) >= int(max_points):
                break
        if len(uniq) == 0:
            return np.zeros((0, 2), dtype=np.int32)
        return np.asarray(uniq, dtype=np.int32)

    def _compute_fixed_bbox(
        self,
        M_pre: Optional[np.ndarray],
        P_pos: Optional[np.ndarray],
        shape: Tuple[int, int],
        size: int,
    ) -> Tuple[int, int, int, int]:
        H, W = shape
        size = int(size)
        if size <= 0:
            size = min(H, W)
        if size > W or size > H:
            return (0, 0, W - 1, H - 1)

        if M_pre is not None and np.any(M_pre > 0):
            ys, xs = np.nonzero(M_pre > 0)
            cx = float(xs.mean())
            cy = float(ys.mean())
        elif P_pos is not None and len(P_pos) > 0:
            cx = float(np.mean(P_pos[:, 0]))
            cy = float(np.mean(P_pos[:, 1]))
        else:
            cx = W / 2.0
            cy = H / 2.0

        half = size // 2
        x0 = int(round(cx - half))
        y0 = int(round(cy - half))
        x1 = x0 + size - 1
        y1 = y0 + size - 1

        if x0 < 0:
            x1 += -x0
            x0 = 0
        if y0 < 0:
            y1 += -y0
            y0 = 0
        if x1 >= W:
            shift = x1 - (W - 1)
            x0 = max(0, x0 - shift)
            x1 = W - 1
        if y1 >= H:
            shift = y1 - (H - 1)
            y0 = max(0, y0 - shift)
            y1 = H - 1

        return (int(x0), int(y0), int(x1), int(y1))

    def _draw_points(self, img_bgr: np.ndarray, points: np.ndarray, color: Tuple[int, int, int], radius: int = 3):
        for x, y in points:
            cv2.circle(img_bgr, (int(x), int(y)), radius, color, -1)

    def _save_voronoi_viz(
        self,
        Iq: Optional[np.ndarray],
        M_pre: np.ndarray,
        label_map: np.ndarray,
        P_sites: List[Tuple[int, int]],
        P_pos: List[List[int]],
        case_id: Optional[str],
        slice_id: Optional[int],
        viz_dir: Optional[str],
    ) -> Optional[str]:
        out_path = self._stage_path(viz_dir, case_id, slice_id, "voronoi")
        if out_path is None:
            return None
        H, W = M_pre.shape
        base = self._to_uint8(Iq if Iq is not None else M_pre.astype(np.float32), normalize=True)
        base = cv2.cvtColor(base, cv2.COLOR_GRAY2BGR)

        label_img = np.zeros((H, W, 3), dtype=np.uint8)
        rng = np.random.RandomState(self.config.seed)
        if label_map is not None:
            for i in range(len(P_sites)):
                color = rng.randint(0, 255, size=3).tolist()
                label_img[label_map == i] = color
        overlay = cv2.addWeighted(base, 1 - self.config.voronoi_viz_alpha, label_img, self.config.voronoi_viz_alpha, 0)

        # Cell boundaries
        boundary = np.zeros((H, W), dtype=np.uint8)
        if label_map is not None:
            boundary[:-1, :] |= (label_map[:-1, :] != label_map[1:, :]).astype(np.uint8)
            boundary[:, :-1] |= (label_map[:, :-1] != label_map[:, 1:]).astype(np.uint8)
        overlay[boundary > 0] = (255, 255, 255)

        # Draw sites and positives
        for x, y in P_sites:
            cv2.circle(overlay, (int(x), int(y)), 2, (0, 0, 255), -1)
        for p in P_pos:
            cv2.circle(overlay, (int(p[0]), int(p[1])), 3, (0, 255, 0), -1)

        cv2.imwrite(out_path, overlay)
        return out_path
