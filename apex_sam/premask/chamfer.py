from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
from scipy.ndimage import binary_dilation, binary_fill_holes, distance_transform_edt
from skimage import measure, morphology
from skimage.segmentation import random_walker


class ChamferMixin:
    def _oriented_chamfer_match(
        self,
        pts: np.ndarray,
        theta: np.ndarray,
        dts: List[np.ndarray],
        rois: List[Tuple[int, int, int, int]],
        dino_mask: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """Search over translation/scale/rotation using oriented chamfer cost."""
        H, W = dts[0].shape
        if pts is None or len(pts) < 8:
            return {"best": None}
        pts = pts.astype(np.float32)
        theta = theta.astype(np.float32)
        c = pts.mean(axis=0)
        pts0 = pts - c[None, :]
        scales = [float(s) for s in self.config.scales]
        rotations = [float(r) for r in self.config.rotations_deg]
        stride = int(self.config.chamfer_stride)
        min_valid = max(4, int(self.config.chamfer_min_valid_ratio * len(pts0)))

        best_cost = 1e9
        best = None
        best_pts = None

        for (x0, y0, x1, y1) in rois:
            for scale in scales:
                pts_s = pts0 * scale
                for rot_deg in rotations:
                    rad = math.radians(rot_deg)
                    cr, sr = math.cos(rad), math.sin(rad)
                    R = np.array([[cr, -sr], [sr, cr]], dtype=np.float32)
                    pts_r = pts_s @ R.T
                    theta_r = theta + rad
                    bin_idx = self._theta_to_bin(theta_r, int(self.config.n_angle_bins))

                    minx = float(pts_r[:, 0].min())
                    maxx = float(pts_r[:, 0].max())
                    miny = float(pts_r[:, 1].min())
                    maxy = float(pts_r[:, 1].max())
                    tx_start = int(math.ceil(max(x0, 0) - minx))
                    tx_end = int(math.floor(min(x1 - 1, W - 1) - maxx))
                    ty_start = int(math.ceil(max(y0, 0) - miny))
                    ty_end = int(math.floor(min(y1 - 1, H - 1) - maxy))
                    if tx_start > tx_end or ty_start > ty_end:
                        continue
                    for ty in range(ty_start, ty_end + 1, stride):
                        for tx in range(tx_start, tx_end + 1, stride):
                            if dino_mask is not None and dino_mask[int(ty), int(tx)] == 0:
                                continue
                            pts_t = pts_r + np.array([tx, ty], dtype=np.float32)
                            xi = np.round(pts_t[:, 0]).astype(np.int32)
                            yi = np.round(pts_t[:, 1]).astype(np.int32)
                            valid = (xi >= 0) & (xi < W) & (yi >= 0) & (yi < H)
                            if valid.sum() < min_valid:
                                continue
                            costs = []
                            idxs = np.where(valid)[0]
                            for ii in idxs:
                                k = int(bin_idx[ii])
                                costs.append(dts[k][yi[ii], xi[ii]])
                            if not costs:
                                continue
                            cost = float(np.mean(costs))
                            if cost < best_cost:
                                best_cost = cost
                                best_pts = pts_t.copy()
                                best = {
                                    "x0": float(tx),
                                    "y0": float(ty),
                                    "scale": float(scale),
                                    "rot_deg": float(rot_deg),
                                    "cost": float(cost),
                                    "roi": (x0, y0, x1, y1),
                                }
        return {"best": best, "best_points": best_pts, "best_cost": best_cost}

    def _premask_random_walker(
        self,
        M0_roi: np.ndarray,
        Eq_roi: np.ndarray,
        fit_mode: str,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Query-native premask extraction with RandomWalker."""
        info: Dict[str, Any] = {"used": False, "fg": None, "bg": None, "band": None}
        if M0_roi.sum() == 0:
            return M0_roi, info
        r_fg = int(self.config.r_fg)
        r_bg = int(self.config.r_bg)
        r_mid = int(self.config.r_mid)
        w_band = int(self.config.w_band)

        if fit_mode == "enclosing":
            fg = morphology.binary_dilation(M0_roi > 0, morphology.disk(max(1, r_fg))).astype(np.uint8)
            bg = morphology.binary_dilation(M0_roi > 0, morphology.disk(max(1, r_bg))).astype(np.uint8)
            bg = np.logical_and(bg > 0, ~morphology.binary_dilation(M0_roi > 0, morphology.disk(max(1, r_mid)))).astype(np.uint8)
        else:
            fg = morphology.binary_erosion(M0_roi > 0, morphology.disk(max(1, r_fg))).astype(np.uint8)
            bg = morphology.binary_dilation(M0_roi > 0, morphology.disk(max(1, r_bg))).astype(np.uint8)
            bg = np.logical_and(bg > 0, ~(M0_roi > 0)).astype(np.uint8)

        D0 = distance_transform_edt(M0_roi > 0) - distance_transform_edt(M0_roi == 0)
        band = (np.abs(D0) <= max(1, w_band)).astype(np.uint8)

        labels = np.zeros_like(M0_roi, dtype=np.int32)
        labels[fg > 0] = 1
        labels[bg > 0] = 2
        labels[(band == 0) & (M0_roi > 0)] = 1
        labels[(band == 0) & (M0_roi == 0)] = 2

        info.update({"fg": fg, "bg": bg, "band": band})

        try:
            rw = random_walker(1.0 - Eq_roi, labels, beta=float(self.config.rw_beta), mode="cg_mg")
            M_pre = (rw == 1).astype(np.uint8)
            info["used"] = True
        except Exception:
            M_pre = M0_roi.astype(np.uint8)
        M_pre = binary_fill_holes(M_pre > 0).astype(np.uint8)
        M_pre = self._keep_largest_component(M_pre).astype(np.uint8)
        return M_pre, info

    def _generate_premask_chamfer(
        self,
        Iq_norm: np.ndarray,
        Is_norm: np.ndarray,
        M_s: np.ndarray,
        Sdino: Optional[np.ndarray],
        dino_region: Optional[np.ndarray] = None,
        support_area: Optional[float] = None,
        support_has_hole: bool = False,
        support_desc: Optional[Dict[str, float]] = None,
    ) -> Dict[str, Any]:
        """Generate query-native pre-mask via oriented chamfer + RW refinement."""
        debug: Dict[str, Any] = {}
        tmpl = self._support_boundary_template(M_s)
        if tmpl is None:
            return {"M_pre": np.zeros_like(Iq_norm, dtype=np.uint8), "match_debug": debug}
        pts = tmpl["points"]
        theta = tmpl["theta"]

        Eq, theta_q, valid = self._compute_structure_maps(Iq_norm)
        debug["Eq"] = Eq

        dino_mask, rois, dino_info = self._dino_gate_rois(dino_region, shape_hw=Iq_norm.shape[:2])
        debug["dino_mask"] = dino_mask
        debug["dino_rois"] = rois
        debug["dino_info"] = dino_info

        dts, _ = self._build_directional_dts(Eq, theta_q, valid)
        match = self._oriented_chamfer_match(pts, theta, dts, rois, dino_mask=dino_mask)
        debug["match"] = match

        best = match.get("best")
        best_pts = match.get("best_points")
        if best is None or best_pts is None:
            return {"M_pre": np.zeros_like(Iq_norm, dtype=np.uint8), "match_debug": debug}

        x0, y0, x1, y1 = best["roi"]
        roi_shape = (y1 - y0, x1 - x0)
        if bool(support_has_hole):
            M0_full = self._warp_support_mask_by_match(
                support_mask=M_s,
                match_best=best,
                center_xy=np.asarray(tmpl.get("center", pts.mean(axis=0)), dtype=np.float32),
                out_shape=Iq_norm.shape[:2],
            )
            M0_roi = (M0_full[y0:y1, x0:x1] > 0).astype(np.uint8)
        else:
            pts_roi = best_pts.copy()
            pts_roi[:, 0] -= x0
            pts_roi[:, 1] -= y0
            M0_roi = self._rasterize_boundary(pts_roi, roi_shape)
        # Optional: use matched structure directly as pre-mask (skip query-native refinement)
        if bool(self.config.premask_matched_only):
            M_pre_roi = M0_roi.astype(np.uint8)
            if not support_has_hole:
                M_pre_roi = binary_fill_holes(M_pre_roi > 0).astype(np.uint8)
            M_pre_roi = self._keep_largest_component(M_pre_roi).astype(np.uint8)
            M_pre = np.zeros_like(Iq_norm, dtype=np.uint8)
            M_pre[y0:y1, x0:x1] = M_pre_roi.astype(np.uint8)
            debug["M0_roi"] = M0_roi
            debug["M_pre"] = M_pre
            debug["best_transform"] = best
            debug["premask_mode"] = "matched_only"
            return {"M_pre": M_pre, "match_debug": debug}
        Eq_roi = Eq[y0:y1, x0:x1]

        M_pre_roi, rw_info = self._premask_random_walker(M0_roi, Eq_roi, self.config.fit_mode)
        debug["rw_info"] = rw_info

        # Enforce coverage against migrated mask (M0)
        migrated_area = float(M0_roi.sum())
        if migrated_area > 0:
            cover = float((M_pre_roi > 0)[M0_roi > 0].sum()) / (migrated_area + 1e-6)
            if cover < float(self.config.premask_min_cover_ratio):
                work = M_pre_roi.copy()
                for _ in range(int(self.config.premask_expand_iters)):
                    cover = float((work > 0)[M0_roi > 0].sum()) / (migrated_area + 1e-6)
                    if cover >= float(self.config.premask_min_cover_ratio):
                        break
                    work = morphology.binary_dilation(work > 0, morphology.disk(1)).astype(np.uint8)
                M_pre_roi = work.astype(np.uint8)
            debug["premask_cover"] = cover

        # Snap to query-native closed structure
        if bool(self.config.closed_refine_enable):
            Iq_roi = Iq_norm[y0:y1, x0:x1]
            Sdino_roi = Sdino[y0:y1, x0:x1] if Sdino is not None else None
            dino_roi = dino_mask[y0:y1, x0:x1] if dino_mask is not None else None
            M_snap, snap_info = self._snap_to_query_closed_structure(
                Eq_roi=Eq_roi,
                M_pre_roi=M_pre_roi,
                support_has_hole=bool(support_has_hole),
                support_area=float(support_area) if support_area is not None else None,
                Sdino_roi=Sdino_roi,
                support_desc=support_desc,
                dino_region_roi=dino_roi,
                migrated_mask_roi=M0_roi,
                Iq_roi=Iq_roi,
            )
            debug["closed_refine"] = snap_info
            if snap_info.get("used"):
                M_pre_roi = M_snap.astype(np.uint8)

        M_pre = np.zeros_like(Iq_norm, dtype=np.uint8)
        M_pre[y0:y1, x0:x1] = M_pre_roi.astype(np.uint8)
        debug["M0_roi"] = M0_roi
        debug["M_pre"] = M_pre
        debug["best_transform"] = best
        return {"M_pre": M_pre, "match_debug": debug}
