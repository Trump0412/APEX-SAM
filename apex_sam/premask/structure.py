from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple, List

import cv2
import numpy as np
from scipy import ndimage
from scipy.ndimage import distance_transform_edt, binary_dilation, binary_fill_holes
from scipy.spatial import cKDTree
from skimage import measure, morphology


class StructureMixin:
    def _theta_to_bin(self, theta: np.ndarray, K: int) -> np.ndarray:
        """Map angles in radians to [0, K-1] bins."""
        bins = ((theta + np.pi) / (2.0 * np.pi) * K).astype(np.int32)
        return np.clip(bins, 0, K - 1)

    def _build_directional_dts(
        self,
        E_q: np.ndarray,
        theta: np.ndarray,
        valid: np.ndarray,
    ) -> Tuple[List[np.ndarray], np.ndarray]:
        """Build directional distance transforms for oriented chamfer matching."""
        K = int(self.config.n_angle_bins)
        bin_idx = self._theta_to_bin(theta, K)
        dts: List[np.ndarray] = []
        for k in range(K):
            mask_k = valid & (bin_idx == k)
            inv = np.ones_like(mask_k, dtype=np.uint8)
            inv[mask_k] = 0
            dts.append(distance_transform_edt(inv))
        return dts, bin_idx

    def _build_dino_region_mask(
        self,
        sim_maps: Optional[np.ndarray],
        Sdino: Optional[np.ndarray],
        sim_mean: Optional[np.ndarray] = None,
    ) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Build high-similarity region mask from maup-style regional similarity maps.
        The mask is derived from top-k regional sim maps, not a single prototype map.
        """
        info: Dict[str, Any] = {"thr": None, "per_region_thr": [], "num_regions": 0, "area": 0}
        if sim_maps is None or sim_maps.size == 0:
            return None, info
        sim_maps = np.asarray(sim_maps)
        if sim_maps.ndim != 3:
            return None, info
        n_regions, H, W = sim_maps.shape
        info["num_regions"] = int(n_regions)

        q = float(self.config.dino_gate_quantile)
        q = max(0.0, min(1.0, q))
        masks = []
        per_thr = []
        for r in range(n_regions):
            sm = sim_maps[r]
            try:
                thr = float(np.quantile(sm, q))
            except Exception:
                thr = None
            if thr is None:
                continue
            per_thr.append(thr)
            masks.append(sm >= thr)
        info["per_region_thr"] = per_thr
        if not masks:
            return None, info

        mask = np.logical_or.reduce(masks).astype(np.uint8)
        # Morphological refinement (reuse roi_dino_region_* controls)
        if self.config.roi_dino_region_dilate > 0:
            mask = morphology.binary_dilation(
                mask > 0, morphology.disk(int(self.config.roi_dino_region_dilate))
            ).astype(np.uint8)
        if self.config.roi_dino_region_close > 0:
            mask = morphology.binary_closing(
                mask > 0, morphology.disk(int(self.config.roi_dino_region_close))
            ).astype(np.uint8)
        if self.config.roi_dino_region_erode > 0:
            mask = morphology.binary_erosion(
                mask > 0, morphology.disk(int(self.config.roi_dino_region_erode))
            ).astype(np.uint8)

        # Keep peak-connected component if requested
        if mask.sum() > 0 and bool(self.config.roi_dino_region_keep_peak_component):
            peak_map = sim_mean
            if peak_map is None and Sdino is not None:
                peak_map = Sdino
            if peak_map is not None:
                try:
                    peak_idx = int(np.argmax(peak_map))
                    py, px = np.unravel_index(peak_idx, peak_map.shape)
                    labels = measure.label(mask > 0, connectivity=1)
                    peak_label = labels[int(py), int(px)]
                    if peak_label > 0:
                        mask = (labels == peak_label).astype(np.uint8)
                    else:
                        mask = self._keep_largest_component(mask).astype(np.uint8)
                except Exception:
                    mask = self._keep_largest_component(mask).astype(np.uint8)

        info["area"] = int(mask.sum())
        return mask.astype(np.uint8), info

    def _dino_gate_rois(
        self,
        dino_region: Optional[np.ndarray],
        shape_hw: Optional[Tuple[int, int]] = None,
    ) -> Tuple[np.ndarray, List[Tuple[int, int, int, int]], Dict[str, Any]]:
        """Build ROI list from DINO regional similarity mask."""
        info: Dict[str, Any] = {"rois": []}
        if not bool(self.config.enable_dino_gate):
            if shape_hw is None:
                return None, [(0, 0, 0, 0)], info
            H, W = int(shape_hw[0]), int(shape_hw[1])
            return np.ones((H, W), dtype=np.uint8), [(0, 0, W, H)], info

        if dino_region is None or dino_region.sum() == 0:
            if shape_hw is None:
                return None, [(0, 0, 0, 0)], info
            H, W = int(shape_hw[0]), int(shape_hw[1])
            return np.ones((H, W), dtype=np.uint8), [(0, 0, W, H)], info

        mask = (dino_region > 0).astype(np.uint8)
        lbl = measure.label(mask, connectivity=1)
        rois: List[Tuple[int, int, int, int]] = []
        H, W = mask.shape
        margin = int(self.config.dino_gate_margin)
        for lab in range(1, lbl.max() + 1):
            ys, xs = np.where(lbl == lab)
            if xs.size == 0:
                continue
            x0 = max(0, int(xs.min()) - margin)
            x1 = min(W, int(xs.max()) + margin + 1)
            y0 = max(0, int(ys.min()) - margin)
            y1 = min(H, int(ys.max()) + margin + 1)
            rois.append((x0, y0, x1, y1))
        if not rois:
            rois = [(0, 0, W, H)]
        info["rois"] = rois
        return mask.astype(np.uint8), rois, info

    def _rasterize_boundary(self, pts: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        """Rasterize boundary points to a filled mask."""
        H, W = shape
        mask = np.zeros((H, W), dtype=np.uint8)
        if pts is None or len(pts) < 3:
            return mask
        poly = np.round(pts).astype(np.int32)
        poly[:, 0] = np.clip(poly[:, 0], 0, W - 1)
        poly[:, 1] = np.clip(poly[:, 1], 0, H - 1)
        cv2.fillPoly(mask, [poly], 1)
        return mask.astype(np.uint8)

    def _warp_support_mask_by_match(
        self,
        support_mask: np.ndarray,
        match_best: Dict[str, Any],
        center_xy: np.ndarray,
        out_shape: Tuple[int, int],
    ) -> np.ndarray:
        """Apply chamfer best rigid transform to support mask (preserve ring topology)."""
        H, W = out_shape
        if support_mask is None or match_best is None or center_xy is None:
            return np.zeros((H, W), dtype=np.uint8)
        s = float(match_best.get("scale", 1.0))
        rot_deg = float(match_best.get("rot_deg", 0.0))
        tx = float(match_best.get("x0", 0.0))
        ty = float(match_best.get("y0", 0.0))
        cx = float(center_xy[0])
        cy = float(center_xy[1])
        rad = math.radians(rot_deg)
        cr, sr = math.cos(rad), math.sin(rad)

        a00 = s * cr
        a01 = -s * sr
        a10 = s * sr
        a11 = s * cr
        b0 = tx - a00 * cx - a01 * cy
        b1 = ty - a10 * cx - a11 * cy
        M = np.array([[a00, a01, b0], [a10, a11, b1]], dtype=np.float32)

        src = (support_mask > 0.5).astype(np.uint8)
        warped = cv2.warpAffine(src, M, (W, H), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return (warped > 0).astype(np.uint8)

    def _shape_match_score(self, pre_desc: Dict[str, float], support_desc: Dict[str, float]) -> float:
        """Score how similar two shapes are using coarse descriptors."""
        supp_area = float(support_desc.get("area", 0.0))
        pre_area = float(pre_desc.get("area", 0.0))
        if supp_area <= 0 or pre_area <= 0:
            return -1e6
        area_ratio = pre_area / supp_area
        area_score = math.exp(-abs(math.log(max(area_ratio, 1e-6))))
        circ_diff = abs(float(pre_desc.get("circularity", 0.0)) - float(support_desc.get("circularity", 0.0)))
        circ_score = max(0.0, 1.0 - circ_diff)
        ecc_diff = abs(float(pre_desc.get("eccentricity", 0.0)) - float(support_desc.get("eccentricity", 0.0)))
        ecc_score = max(0.0, 1.0 - ecc_diff)
        hole_match = float(pre_desc.get("has_hole", 0.0)) == float(support_desc.get("has_hole", 0.0))
        hole_bias = 0.1 if hole_match else -0.1
        return 0.5 * area_score + 0.3 * circ_score + 0.2 * ecc_score + hole_bias

    def _snap_to_query_closed_structure(
        self,
        Eq_roi: np.ndarray,
        M_pre_roi: np.ndarray,
        support_has_hole: bool,
        support_area: Optional[float],
        Sdino_roi: Optional[np.ndarray] = None,
        support_desc: Optional[Dict[str, float]] = None,
        dino_region_roi: Optional[np.ndarray] = None,
        migrated_mask_roi: Optional[np.ndarray] = None,
        Iq_roi: Optional[np.ndarray] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        将 pre-mask 吸附到查询边缘图中的最近闭合结构，保证 query-native。
        """
        H, W = Eq_roi.shape
        info: Dict[str, Any] = {
            "used": False,
            "thr": None,
            "cand_count": 0,
            "best_score": None,
            "best_mean_dist": None,
            "best_has_hole": None,
            "edge_bin": None,
            "support_area": float(support_area) if support_area is not None else None,
            "area_min": None,
            "area_max": None,
            "area": None,
            "area_ratio": None,
            "area_clamped": False,
            "dino_thr": None,
            "dino_area": None,
            "dino_region_area": None,
            "dino_region_cover_ratio": None,
            "dino_region_cover_req": float(self.config.dino_region_min_cover_ratio),
            "migrated_cover_ratio": None,
            "migrated_cover_req": float(self.config.premask_min_cover_ratio),
        }
        if H < 4 or W < 4 or M_pre_roi.sum() == 0 or support_area is None or support_area <= 0:
            return M_pre_roi, info

        min_area_bound = float(self.config.closed_area_min_ratio) * float(support_area)
        max_area_bound = float(self.config.closed_area_max_ratio) * float(support_area)
        info["area_min"] = min_area_bound
        info["area_max"] = max_area_bound
        min_area_geom = float(self.config.closed_min_area_ratio) * float(H * W)
        pre_d = morphology.binary_dilation(M_pre_roi > 0, morphology.disk(int(self.config.closed_pre_dilate)))
        pre_d = pre_d.astype(np.uint8)
        pre_mask = pre_d > 0
        pre_boundary = self._mask_boundary(M_pre_roi) > 0
        if not pre_boundary.any():
            pre_boundary = pre_mask

        region_mask = None
        region_area = 0.0
        if dino_region_roi is not None and np.any(dino_region_roi > 0):
            region_mask = (dino_region_roi > 0)
            if self.config.roi_dino_region_dilate > 0:
                region_mask = morphology.binary_dilation(
                    region_mask, morphology.disk(int(self.config.roi_dino_region_dilate))
                )
            region_mask = region_mask.astype(np.uint8)
            region_area = float(region_mask.sum())
            info["dino_region_area"] = region_area

        migrated_mask = None
        migrated_area = 0.0
        if migrated_mask_roi is not None and np.any(migrated_mask_roi > 0):
            migrated_mask = (migrated_mask_roi > 0)
            migrated_area = float(migrated_mask.sum())

        dino_candidate: Optional[np.ndarray] = None
        dino_expand_candidate: Optional[np.ndarray] = None
        if Sdino_roi is not None and bool(self.config.closed_dino_enable):
            q_dino = float(self.config.closed_dino_quantile)
            q_dino = max(0.0, min(1.0, q_dino))
            try:
                thr_dino = float(np.quantile(Sdino_roi, q_dino))
                dmask = (Sdino_roi >= thr_dino).astype(np.uint8)
                if self.config.closed_dino_dilate > 0:
                    dmask = morphology.binary_dilation(dmask > 0, morphology.disk(int(self.config.closed_dino_dilate))).astype(np.uint8)
                if self.config.closed_dino_close > 0:
                    dmask = morphology.binary_closing(dmask > 0, morphology.disk(int(self.config.closed_dino_close))).astype(np.uint8)
                if not support_has_hole:
                    dmask = binary_fill_holes(dmask > 0).astype(np.uint8)
                dmask = self._keep_largest_component(dmask).astype(np.uint8)
                if dmask.sum() > 0:
                    dino_candidate = dmask.astype(np.uint8)
                    info["dino_thr"] = thr_dino
                    info["dino_area"] = float(dmask.sum())
            except Exception:
                dino_candidate = None

        dino_mask_bool = (dino_candidate > 0) if dino_candidate is not None else None
        dino_mask_area = float(dino_mask_bool.sum()) if dino_mask_bool is not None else 0.0

        if (
            dino_mask_bool is not None
            and bool(self.config.closed_dino_expand_enable)
            and support_area is not None
            and support_area > 0
        ):
            gate = morphology.binary_dilation(dino_mask_bool, morphology.disk(int(self.config.closed_dino_expand_gate_dilate)))
            target_ratio = float(self.config.closed_dino_expand_target_ratio)
            target_area = float(np.clip(target_ratio, 0.5, 1.5) * float(support_area))
            work = dino_mask_bool.astype(np.uint8)
            for _ in range(64):
                area = float(work.sum())
                if area >= target_area:
                    break
                work = morphology.binary_dilation(work > 0, morphology.disk(1)).astype(np.uint8)
                work = np.logical_and(work > 0, gate > 0).astype(np.uint8)
            if not support_has_hole:
                work = binary_fill_holes(work > 0).astype(np.uint8)
            work = self._keep_largest_component(work).astype(np.uint8)
            if work.sum() > 0:
                dino_expand_candidate = work.astype(np.uint8)

        Eq_focus = Eq_roi
        if Sdino_roi is not None and bool(self.config.closed_dino_edge_enable):
            Sd_focus = self._normalize(np.clip(Sdino_roi.astype(np.float32), 0.0, 1.0))
            gamma = max(0.5, float(self.config.closed_dino_edge_gamma))
            Sd_focus = np.power(Sd_focus, gamma)
            w_edge = float(self.config.closed_dino_edge_weight)
            w_edge = max(0.0, min(1.0, w_edge))
            Eq_focus = Eq_roi * ((1.0 - w_edge) + w_edge * Sd_focus)
            Eq_focus = self._normalize(Eq_focus.astype(np.float32))

        if bool(self.config.dino_hf_match_enable) and Iq_roi is not None:
            hf = cv2.Laplacian(Iq_roi.astype(np.float32), cv2.CV_32F)
            hf = np.abs(hf)
            hf = self._normalize(hf)
            if region_mask is not None:
                hf = hf * region_mask.astype(np.float32)
            w_hf = max(0.0, min(1.0, float(self.config.dino_hf_weight)))
            Eq_focus = Eq_focus * ((1.0 - w_hf) + w_hf * hf)
            Eq_focus = self._normalize(Eq_focus.astype(np.float32))

        if region_mask is not None:
            Eq_focus = Eq_focus * region_mask.astype(np.float32)

        diag = math.sqrt(float(H * H + W * W))
        max_mean_dist = float(self.config.closed_max_mean_dist_frac) * diag
        ignore_area_bounds = bool(self.config.dino_region_ignore_support_area) and region_area > 0

        def _area_clamp(mask: np.ndarray) -> Tuple[np.ndarray, bool]:
            if ignore_area_bounds:
                return (mask > 0).astype(np.uint8), False
            out = (mask > 0).astype(np.uint8)
            area = float(out.sum())
            clamped = False
            if area > max_area_bound:
                # If the mask is extremely large, try restricting to the pre-mask neighborhood.
                overlap = float((out > 0).astype(np.uint8)[pre_mask].sum())
                if overlap > 0 and area > 2.5 * max_area_bound:
                    restricted = np.logical_and(out > 0, pre_mask).astype(np.uint8)
                    if restricted.sum() >= min_area_bound:
                        out = restricted
                        area = float(out.sum())
                        clamped = True
                erosion_iters = 0
                while area > max_area_bound and area > 0 and erosion_iters < 32:
                    out = morphology.binary_erosion(out > 0, morphology.disk(1)).astype(np.uint8)
                    area = float(out.sum())
                    erosion_iters += 1
                clamped = clamped or (erosion_iters > 0)
                # If we eroded too much, dilate back toward the lower bound.
                dilate_back_iters = 0
                while area < min_area_bound and dilate_back_iters < 8:
                    out = morphology.binary_dilation(out > 0, morphology.disk(1)).astype(np.uint8)
                    area = float(out.sum())
                    dilate_back_iters += 1
                clamped = clamped or (dilate_back_iters > 0)
            elif area < min_area_bound:
                overlap = float(((out > 0) & pre_mask).sum())
                overlap_ratio = overlap / max(1.0, area)
                if overlap_ratio >= 0.1:
                    out = np.logical_or(out > 0, pre_mask).astype(np.uint8)
                    area = float(out.sum())
                clamped = True
                dilate_iters = 0
                while area < min_area_bound and dilate_iters < 8:
                    out = morphology.binary_dilation(out > 0, morphology.disk(1)).astype(np.uint8)
                    area = float(out.sum())
                    dilate_iters += 1
                # Avoid exceeding the upper bound after dilation.
                erode_back_iters = 0
                while area > max_area_bound and erode_back_iters < 8:
                    out = morphology.binary_erosion(out > 0, morphology.disk(1)).astype(np.uint8)
                    area = float(out.sum())
                    erode_back_iters += 1
            if out.sum() == 0:
                return (mask > 0).astype(np.uint8), False
            return out.astype(np.uint8), clamped

        def _candidates_from_edges(edge_bin: np.ndarray) -> List[np.ndarray]:
            contours, hierarchy = cv2.findContours((edge_bin * 255).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
            cands: List[np.ndarray] = []
            if hierarchy is None or len(contours) == 0:
                return cands
            hierarchy0 = hierarchy[0]
            for idx, h in enumerate(hierarchy0):
                if h[3] != -1:
                    continue
                cand = np.zeros((H, W), dtype=np.uint8)
                cv2.drawContours(cand, contours, idx, color=1, thickness=-1)
                child = h[2]
                while child != -1:
                    cv2.drawContours(cand, contours, child, color=0, thickness=-1)
                    child = hierarchy0[child][0]
                if cand.sum() > 0:
                    cands.append(cand.astype(np.uint8))
            return cands

        def _select_best(candidates: List[np.ndarray]) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
            best_any = None
            best_any_score = -1e9
            best_any_mean_dist = None
            best_any_has_hole = None
            best_any_area = None
            best_cover = None
            best_cover_ratio = -1.0
            best_in_bounds = None
            best_in_bounds_score = -1e9
            stats = {"valid_count": 0}
            for cand in candidates:
                area = float(cand.sum())
                if area < min_area_geom:
                    continue
                stats["valid_count"] += 1
                cand_bool = cand > 0
                overlap = float((cand_bool & pre_mask).sum())
                pre_area = float(pre_mask.sum()) + 1e-6
                cand_area = float(cand_bool.sum()) + 1e-6
                overlap_score = 0.5 * (overlap / pre_area) + 0.5 * (overlap / cand_area)

                cover_ratio = None
                if migrated_area > 0:
                    cover_ratio = float((cand_bool & migrated_mask).sum()) / (migrated_area + 1e-6)
                    if cover_ratio > best_cover_ratio:
                        best_cover_ratio = cover_ratio
                        best_cover = cand.astype(np.uint8)
                    if cover_ratio < float(self.config.premask_min_cover_ratio):
                        continue

                dist_map = distance_transform_edt(~cand_bool)
                mean_dist = float(dist_map[pre_boundary].mean())
                dist_score = 1.0 - min(1.0, mean_dist / (max_mean_dist + 1e-6))

                filled = binary_fill_holes(cand_bool).astype(np.uint8)
                has_hole = bool(filled.sum() > cand_bool.sum())
                hole_factor = 1.0
                if support_has_hole and has_hole:
                    hole_factor += float(self.config.closed_hole_bonus)
                if (not support_has_hole) and has_hole:
                    hole_factor -= float(self.config.closed_hole_penalty)

                base_score = (0.6 * overlap_score + 0.4 * dist_score) * hole_factor
                if dino_mask_bool is not None and dino_mask_area > 0:
                    inter_dm = float((cand_bool & dino_mask_bool).sum())
                    dino_mask_score = 0.5 * (inter_dm / (dino_mask_area + 1e-6)) + 0.5 * (inter_dm / cand_area)
                    w_dm = float(self.config.closed_dino_mask_weight)
                    w_dm = max(0.0, min(1.0, w_dm))
                    base_score = (1.0 - w_dm) * base_score + w_dm * dino_mask_score
                if support_desc is not None:
                    cand_desc = self._mask_descriptors(cand_bool.astype(np.uint8))
                    shape_score = self._shape_match_score(cand_desc, support_desc)
                    w_shape = float(self.config.closed_shape_weight)
                    w_shape = max(0.0, min(1.0, w_shape))
                    base_score = (1.0 - w_shape) * base_score + w_shape * shape_score
                score = base_score
                if Sdino_roi is not None:
                    vals = Sdino_roi[cand_bool]
                    if vals.size > 0:
                        dino_mean = float(vals.mean())
                        w = float(self.config.closed_dino_weight)
                        w = max(0.0, min(1.0, w))
                        score = (1.0 - w) * base_score + w * dino_mean
                if cover_ratio is not None:
                    score = score + 0.3 * float(cover_ratio)
                if mean_dist > max_mean_dist:
                    score -= 1.0

                if score > best_any_score:
                    best_any_score = score
                    best_any = cand.astype(np.uint8)
                    best_any_mean_dist = mean_dist
                    best_any_has_hole = has_hole
                    best_any_area = area

                in_bounds = True
                if not ignore_area_bounds:
                    in_bounds = (min_area_bound <= area <= max_area_bound)
                if in_bounds and score > best_in_bounds_score:
                    best_in_bounds_score = score
                    best_in_bounds = cand.astype(np.uint8)
            stats.update({
                "best_any_score": best_any_score if stats["valid_count"] > 0 else None,
                "best_any_mean_dist": best_any_mean_dist,
                "best_any_has_hole": best_any_has_hole,
                "best_any_area": best_any_area,
                "best_in_bounds_score": best_in_bounds_score if best_in_bounds is not None else None,
                "best_cover_ratio": best_cover_ratio if best_cover is not None else None,
            })
            if best_in_bounds is None and best_cover is not None:
                stats["best_cover_used"] = True
                return best_cover, None, stats
            return best_any, best_in_bounds, stats

        q_cur = float(self.config.closed_edge_quantile)
        q_step = float(self.config.closed_quantile_step)
        q_iters = max(1, int(self.config.closed_quantile_iters))
        best_overall = None
        best_overall_stats = None
        best_overall_edge = None
        best_overall_thr = None
        for _ in range(q_iters):
            q_cur = max(0.55, min(0.98, q_cur))
            thr = float(np.quantile(Eq_focus, q_cur))
            edge_bin = (Eq_focus >= thr).astype(np.uint8)
            if self.config.closed_edge_dilate > 0:
                edge_bin = morphology.binary_dilation(edge_bin, morphology.disk(int(self.config.closed_edge_dilate))).astype(np.uint8)
            if self.config.closed_edge_close > 0:
                edge_bin = morphology.binary_closing(edge_bin, morphology.disk(int(self.config.closed_edge_close))).astype(np.uint8)
            candidates = _candidates_from_edges(edge_bin)
            if dino_candidate is not None:
                candidates = candidates + [dino_candidate]
            if dino_expand_candidate is not None:
                candidates = candidates + [dino_expand_candidate]
            if not candidates:
                q_cur += q_step
                continue
            best_any, best_in_bounds, stats = _select_best(candidates)
            if best_in_bounds is not None:
                out = best_in_bounds.astype(np.uint8)
                if not support_has_hole:
                    out = binary_fill_holes(out > 0).astype(np.uint8)
                out = self._keep_largest_component(out).astype(np.uint8)
                # Track migrated-mask coverage
                if migrated_area > 0:
                    cover_ratio = float((out > 0).astype(np.uint8)[migrated_mask > 0].sum()) / (migrated_area + 1e-6)
                    info["migrated_cover_ratio"] = cover_ratio
                out, clamped_extra = _area_clamp(out)
                info.update({
                    "used": True,
                    "thr": thr,
                    "cand_count": stats.get("valid_count", 0),
                    "best_score": stats.get("best_in_bounds_score"),
                    "best_mean_dist": stats.get("best_any_mean_dist"),
                    "best_has_hole": stats.get("best_any_has_hole"),
                    "edge_bin": edge_bin,
                    "area_clamped": bool(clamped_extra),
                })
                area = float(out.sum())
                info["area"] = area
                info["area_ratio"] = area / float(support_area)
                return out, info

            if best_any is not None:
                best_overall = best_any
                best_overall_stats = stats
                best_overall_edge = edge_bin
                best_overall_thr = thr
                best_area = stats.get("best_any_area")
                if best_area is not None:
                    if best_area > max_area_bound:
                        q_cur += q_step
                    elif best_area < min_area_bound:
                        q_cur -= q_step
                    else:
                        break
                else:
                    q_cur += q_step

        if best_overall is None:
            return M_pre_roi, info

        out = best_overall.astype(np.uint8)
        out, clamped = _area_clamp(out)
        if not support_has_hole:
            out = binary_fill_holes(out > 0).astype(np.uint8)
        out = self._keep_largest_component(out).astype(np.uint8)
        if migrated_area > 0:
            cover_ratio = float((out > 0).astype(np.uint8)[migrated_mask > 0].sum()) / (migrated_area + 1e-6)
            info["migrated_cover_ratio"] = cover_ratio
        out, clamped_extra = _area_clamp(out)
        clamped = bool(clamped or clamped_extra)
        info.update({
            "used": True,
            "thr": best_overall_thr,
            "cand_count": best_overall_stats.get("valid_count", 0) if best_overall_stats else 0,
            "best_score": best_overall_stats.get("best_any_score") if best_overall_stats else None,
            "best_mean_dist": best_overall_stats.get("best_any_mean_dist") if best_overall_stats else None,
            "best_has_hole": best_overall_stats.get("best_any_has_hole") if best_overall_stats else None,
            "edge_bin": best_overall_edge,
            "area_clamped": bool(clamped),
        })
        area = float(out.sum())
        info["area"] = area
        info["area_ratio"] = area / float(support_area)
        return out.astype(np.uint8), info

    def _compute_shape_priors(self, Dalign: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        A5. 计算形状先验

        Args:
            Dalign: 对齐后的距离场 (H, W)

        Returns:
            Pin: 内部先验 (H, W), [0, 1]
            Pband: 边界带先验 (H, W), [0, 1]
        """
        # Inside prior: sigmoid
        Pin = 1.0 / (1.0 + np.exp(-Dalign / self.config.tau_in))

        # Band prior: exponential falloff
        Pband = np.exp(-np.abs(Dalign) / self.config.tau_band)

        return Pin, Pband
