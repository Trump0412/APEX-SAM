from __future__ import annotations

import math
import logging
import os
from typing import Any, Dict, Optional, Tuple, List

import cv2
import numpy as np
import torch
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from skimage import measure

from apex_sam.config import ApexConfig
from apex_sam.hmf.fusion import BranchPrediction, HMFSimpleFusion
from apex_sam.premask.chamfer import ChamferMixin
from apex_sam.premask.edges import EdgeMixin
from apex_sam.premask.structure import StructureMixin
from apex_sam.prompting.voronoi import VoronoiPromptMixin
from apex_sam.retrieval.dino_encoder import DinoFeatureMixin
from apex_sam.sam.predictor import SamMixin
from apex_sam.types import PredictionResult


class ApexSegmenter(DinoFeatureMixin, EdgeMixin, StructureMixin, ChamferMixin, VoronoiPromptMixin, SamMixin):
    def __init__(self, config: Optional[ApexConfig] = None):
        """
        初始化分割器

        Args:
            config: 配置对象,如果为None则使用默认配置
        """
        self.config = config if config is not None else ApexConfig()

        # 设置随机种子
        self._set_seed(self.config.seed)

        # 设备
        self.device = torch.device(self.config.device)

        # 加载模型
        print("[APEX-SAM] Initializing...")
        print(f"[APEX-SAM] Device: {self.device}")

        # 加载 DINOv3
        self.dino_model = self._load_dinov3()

        # 加载 SAM2
        self.sam2_model = self._load_sam2()

        self.hmf = HMFSimpleFusion(
            temperature=float(self.config.hmf_temperature),
            clip_eps=float(self.config.hmf_clip_eps),
            prior_bias=float(self.config.hmf_prior_bias),
        )

        print("[APEX-SAM] Ready")

    def _set_seed(self, seed: int):
        """设置随机种子以保证可复现性"""
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _preprocess_image(self, img: np.ndarray) -> np.ndarray:
        """
        A1. 预处理图像

        Args:
            img: 输入图像 (H, W) 或 (H, W, C)

        Returns:
            归一化后的图像 [0, 1]
        """
        img = img.astype(np.float32)

        # 如果是多通道,取平均
        if img.ndim == 3:
            img = img.mean(axis=-1)

        # Clip to percentiles
        p1, p99 = np.percentile(img, [self.config.p1, self.config.p99])
        img = np.clip(img, p1, p99)

        # Normalize to [0, 1]
        if p99 - p1 > 1e-6:
            img = (img - p1) / (p99 - p1)
        else:
            img = np.zeros_like(img)

        return img

    def _keep_largest_component(self, mask: np.ndarray) -> np.ndarray:
        """
        仅保留掩码中最大的连通区域，去除离散小块
        """
        mask_bin = (mask > 0.5).astype(np.uint8)
        labels = measure.label(mask_bin, connectivity=1)
        if labels.max() == 0:
            return mask_bin
        counts = np.bincount(labels.flatten())
        counts[0] = 0
        largest_label = int(np.argmax(counts))
        cleaned = (labels == largest_label).astype(mask.dtype)
        return cleaned

    def _crop_to_content(self, img: np.ndarray, mask: Optional[np.ndarray] = None,
                         margin: int = 2, thresh_ratio: float = 0.01) -> Tuple[np.ndarray, Optional[np.ndarray], Tuple[int, int, int, int]]:
        """
        去除四周黑边，返回裁剪后的图像/掩码及 bbox

        Args:
            img: 输入图像 (H, W)
            mask: 可选掩码 (H, W)，若提供则优先用掩码定位前景
            margin: 额外保留的边界像素
            thresh_ratio: 无掩码时用于判断前景的强度比例阈值

        Returns:
            img_crop, mask_crop, bbox(x0, y0, x1, y1)
        """
        H, W = img.shape[:2]
        if mask is not None and mask.any():
            ys, xs = np.nonzero(mask > 0.5)
        else:
            thr = img.max() * thresh_ratio
            ys, xs = np.nonzero(img > thr)

        if len(xs) == 0 or len(ys) == 0:
            return img, mask, (0, 0, W, H)

        x0 = max(0, xs.min() - margin)
        x1 = min(W, xs.max() + margin + 1)
        y0 = max(0, ys.min() - margin)
        y1 = min(H, ys.max() + margin + 1)

        img_c = img[y0:y1, x0:x1]
        mask_c = mask[y0:y1, x0:x1] if mask is not None else None

        return img_c, mask_c, (x0, y0, x1, y1)

    def _bbox_from_mask(self, mask: np.ndarray) -> Tuple[int, int, int, int]:
        """Tight bbox around mask; returns full image if empty."""
        H, W = mask.shape[:2]
        ys, xs = np.nonzero(mask > 0)
        if xs.size == 0 or ys.size == 0:
            return 0, 0, W, H
        x0 = int(xs.min())
        x1 = int(xs.max()) + 1
        y0 = int(ys.min())
        y1 = int(ys.max()) + 1
        return x0, y0, x1, y1

    def _compute_signed_distance(self, mask: np.ndarray) -> np.ndarray:
        """
        计算二值掩码的有符号距离场 (H, W).
        """
        mask_bin = (mask > 0.5).astype(np.uint8)
        Din = distance_transform_edt(mask_bin)
        Dout = distance_transform_edt(1 - mask_bin)
        return (Din - Dout).astype(np.float32)

    def _mask_descriptors(self, mask: np.ndarray) -> Dict[str, float]:
        """Compute simple, scale-invariant shape descriptors."""
        mask_bin = (mask > 0).astype(np.uint8)
        area = float(mask_bin.sum())
        if area <= 0:
            return {
                "area": 0.0,
                "perimeter": 0.0,
                "circularity": 0.0,
                "eccentricity": 0.0,
                "has_hole": 0.0,
            }
        boundary = (self._mask_boundary(mask_bin) > 0).astype(np.uint8)
        perimeter = float(boundary.sum())
        if perimeter <= 0:
            circularity = 0.0
        else:
            circularity = float(4.0 * math.pi * area / (perimeter * perimeter + 1e-6))
        lbl = measure.label(mask_bin, connectivity=1)
        props = measure.regionprops(lbl)
        eccentricity = float(props[0].eccentricity) if props else 0.0
        filled = binary_fill_holes(mask_bin > 0).astype(np.uint8)
        has_hole = 1.0 if filled.sum() > mask_bin.sum() else 0.0
        return {
            "area": area,
            "perimeter": perimeter,
            "circularity": circularity,
            "eccentricity": eccentricity,
            "has_hole": has_hole,
        }

    def _run_sam_with_points(
        self,
        bundle: Dict[str, Any],
        P_pos: np.ndarray,
        P_neg: np.ndarray,
        case_id: Optional[str] = None,
        slice_id: Optional[int] = None,
        logger: Optional[logging.Logger] = None,
        gt_mask: Optional[np.ndarray] = None,
        verbose_log: bool = True,
        debug_dict: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        if logger is not None and verbose_log:
            log = logger.info
        elif logger is None and verbose_log:
            log = print
        else:
            log = lambda *args, **kwargs: None

        if debug_dict is None:
            debug_dict = {"viz_paths": []}

        Iq_norm = bundle["Iq_norm"]
        M_pre = bundle["M_pre"]
        viz_task0 = bundle.get("viz_task0")
        viz_task2 = bundle.get("viz_task2")

        P_pos = self._dedup_points(P_pos)
        P_neg = self._dedup_points(P_neg)
        debug_dict["Ppos"] = P_pos
        debug_dict["Pneg"] = P_neg

        if self.config.save_debug_viz:
            p_pts = self._stage_path(viz_task0, case_id, slice_id, "points")
            if p_pts:
                pts_overlay = cv2.cvtColor(self._to_uint8(Iq_norm, normalize=True), cv2.COLOR_GRAY2BGR)
                if len(P_pos) > 0:
                    self._draw_points(pts_overlay, P_pos, color=(0, 255, 0), radius=3)
                if len(P_neg) > 0:
                    self._draw_points(pts_overlay, P_neg, color=(0, 0, 255), radius=3)
                cv2.imwrite(p_pts, pts_overlay)
                debug_dict["viz_paths"].append(p_pts)

        bbox = self._compute_fixed_bbox(M_pre, P_pos, Iq_norm.shape, self.config.bbox_size)
        debug_dict["bbox"] = bbox
        if self.config.save_debug_viz:
            x0, y0, x1, y1 = bbox
            bbox_overlay = cv2.cvtColor(self._to_uint8(Iq_norm, normalize=True), cv2.COLOR_GRAY2BGR)
            cv2.rectangle(bbox_overlay, (x0, y0), (x1, y1), (255, 0, 255), 1)
            p_bbox = self._stage_path(viz_task2, case_id, slice_id, "bbox")
            if p_bbox:
                cv2.imwrite(p_bbox, bbox_overlay)
                debug_dict["viz_paths"].append(p_bbox)

        log("  [B1] SAM2 推理...")
        masks, sam_scores, F_sam = self._run_sam2(Iq_norm, P_pos, P_neg, bbox=bbox)
        debug_dict["masks_initial"] = masks
        debug_dict["sam_scores"] = sam_scores

        best_idx = int(np.argmax(sam_scores)) if sam_scores else 0
        M_final = masks[best_idx]
        M_final = self._keep_largest_component((M_final > 0.5).astype(np.uint8)).astype(np.uint8)
        debug_dict["sam_choice"] = {"index": best_idx, "score": sam_scores[best_idx] if sam_scores else None}
        debug_dict["M_final"] = M_final

        if self.config.save_debug_viz and gt_mask is not None:
            p_cmp = self._stage_path(viz_task0, case_id, slice_id, "pred_vs_gt")
            if p_cmp:
                gt_bin = (gt_mask > 0.5).astype(np.uint8)
                pred_bin = (M_final > 0.5).astype(np.uint8)
                debug_dict["viz_paths"].append(self._save_pred_gt_compare(Iq_norm, pred_bin, gt_bin, p_cmp))
            p_pred = self._stage_path(viz_task0, case_id, slice_id, "pred_mask")
            if p_pred:
                pred_overlay = self._overlay_mask(Iq_norm, (M_final > 0.5).astype(np.uint8), color=(0, 255, 0), alpha=0.4)
                cv2.imwrite(p_pred, pred_overlay)
                debug_dict["viz_paths"].append(p_pred)
            p_diff = self._stage_path(viz_task0, case_id, slice_id, "pred_vs_gt_color")
            if p_diff:
                gt_bin = (gt_mask > 0.5).astype(np.uint8)
                pred_bin = (M_final > 0.5).astype(np.uint8)
                tp = (pred_bin == 1) & (gt_bin == 1)
                fp = (pred_bin == 1) & (gt_bin == 0)
                fn = (pred_bin == 0) & (gt_bin == 1)
                base = cv2.cvtColor(self._to_uint8(Iq_norm, normalize=True), cv2.COLOR_GRAY2BGR)
                base[tp] = (0, 255, 0)
                base[fp] = (0, 0, 255)
                base[fn] = (255, 0, 0)
                cv2.imwrite(p_diff, base)
                debug_dict["viz_paths"].append(p_diff)

        return M_final, debug_dict

    @staticmethod
    def _pick_best_mask_prob(masks: List[np.ndarray], scores: List[float], shape: Tuple[int, int]) -> Tuple[np.ndarray, float]:
        if masks is None or len(masks) == 0:
            return np.zeros(shape, dtype=np.float32), 0.0
        index = int(np.argmax(scores)) if scores else 0
        index = int(np.clip(index, 0, len(masks) - 1))
        prob = np.asarray(masks[index], dtype=np.float32)
        if prob.shape != shape:
            prob = cv2.resize(prob, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
        score = float(scores[index]) if scores else 0.0
        return np.clip(prob, 0.0, 1.0), score

    def _prior_branch_confidence(self, prior_mask: np.ndarray, sdino: Optional[np.ndarray]) -> float:
        prior = (prior_mask > 0.5)
        if prior.sum() <= 0:
            return 0.0
        if sdino is None:
            return 0.5
        q = float(np.clip(self.config.dino_gate_quantile, 0.5, 0.99))
        thr = float(np.quantile(sdino, q))
        dino_hi = sdino >= thr
        overlap = float((prior & dino_hi).sum()) / float(prior.sum() + 1e-6)
        return float(np.clip(overlap, 0.0, 1.0))

    def _run_hmf_branches(
        self,
        bundle: Dict[str, Any],
        P_pos: np.ndarray,
        P_neg: np.ndarray,
        debug_dict: Dict[str, Any],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        Iq_norm = bundle["Iq_norm"]
        M_pre = bundle["M_pre"]
        h, w = Iq_norm.shape[:2]
        bbox = self._compute_fixed_bbox(M_pre, P_pos, Iq_norm.shape, self.config.bbox_size)
        debug_dict["bbox"] = bbox

        point_masks, point_scores, _ = self._run_sam2(Iq_norm, P_pos, P_neg, bbox=None)
        point_prob, point_conf = self._pick_best_mask_prob(point_masks, point_scores, (h, w))

        box_masks, box_scores, _ = self._run_sam_box(Iq_norm, bbox)
        box_prob, box_conf = self._pick_best_mask_prob(box_masks, box_scores, (h, w))

        prior_prob = (M_pre > 0.5).astype(np.float32)
        prior_conf = self._prior_branch_confidence(prior_prob, bundle.get("Sdino"))

        branch_data = [
            BranchPrediction(name="point", prob=point_prob, confidence=float(point_conf)),
            BranchPrediction(name="box", prob=box_prob, confidence=float(box_conf)),
            BranchPrediction(name="prior", prob=prior_prob, confidence=float(prior_conf)),
        ]

        fused_mask, fusion_debug = self.hmf.fuse(branch_data)
        fused_mask = self._keep_largest_component(fused_mask.astype(np.uint8)).astype(np.uint8)
        debug_dict["hmf"] = fusion_debug
        debug_dict["hmf_branch_scores"] = {
            "point": float(point_conf),
            "box": float(box_conf),
            "prior": float(prior_conf),
        }
        return fused_mask, debug_dict

    def _normalize(self, arr: np.ndarray) -> np.ndarray:
        """归一化数组到 [0, 1]"""
        arr_min = arr.min()
        arr_max = arr.max()
        if arr_max - arr_min > 1e-8:
            return (arr - arr_min) / (arr_max - arr_min)
        return np.zeros_like(arr)

    def _ensure_dir(self, path: str):
        if path is None:
            return
        os.makedirs(path, exist_ok=True)

    def _format_case_slice(self, case_id: Optional[str], slice_id: Optional[int]) -> Tuple[str, str]:
        case_str = str(case_id) if case_id is not None else "caseNA"
        slice_str = str(slice_id) if slice_id is not None else "sliceNA"
        return case_str, slice_str

    def _viz_stage_allowed(self, stage: str) -> bool:
        if not bool(getattr(self.config, "viz_minimal_only", False)):
            return True
        keep = getattr(self.config, "viz_keep_stages", None)
        if keep is None:
            return True
        return stage in keep

    def _stage_path(self, viz_dir: Optional[str], case_id: Optional[str], slice_id: Optional[int], stage: str) -> Optional[str]:
        if viz_dir is None:
            return None
        if not self._viz_stage_allowed(stage):
            return None
        self._ensure_dir(viz_dir)
        case_str, slice_str = self._format_case_slice(case_id, slice_id)
        fname = f"{case_str}_slice{slice_str}_{stage}.png"
        return os.path.join(viz_dir, fname)

    def _to_uint8(self, arr: np.ndarray, normalize: bool = True) -> np.ndarray:
        img = arr.astype(np.float32)
        if normalize:
            img = self._normalize(img)
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        return img

    def _save_gray(self, arr: np.ndarray, out_path: Optional[str], normalize: bool = True, colormap: Optional[int] = None) -> Optional[str]:
        if out_path is None:
            return None
        img_u8 = self._to_uint8(arr, normalize=normalize)
        if colormap is not None:
            img_u8 = cv2.applyColorMap(img_u8, colormap)
        else:
            img_u8 = cv2.cvtColor(img_u8, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(out_path, img_u8)
        return out_path

    def _mask_boundary(self, mask: np.ndarray) -> np.ndarray:
        mask_u8 = (mask > 0).astype(np.uint8)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        boundary = cv2.morphologyEx(mask_u8, cv2.MORPH_GRADIENT, kernel)
        return boundary

    def _overlay_boundary(self, img: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        if img.ndim == 2:
            base = cv2.cvtColor(self._to_uint8(img, normalize=True), cv2.COLOR_GRAY2BGR)
        else:
            base = img.copy()
        boundary = self._mask_boundary(mask)
        base[boundary > 0] = color
        return base

    def _overlay_mask(self, img: np.ndarray, mask: np.ndarray, color: Tuple[int, int, int] = (0, 255, 0),
                      alpha: float = 0.4) -> np.ndarray:
        if img.ndim == 2:
            base = cv2.cvtColor(self._to_uint8(img, normalize=True), cv2.COLOR_GRAY2BGR)
        else:
            base = img.copy()
        overlay = base.copy()
        overlay[mask > 0] = color
        return cv2.addWeighted(overlay, alpha, base, 1 - alpha, 0)

    def _save_pred_gt_compare(self, img: np.ndarray, pred: np.ndarray, gt: np.ndarray,
                              out_path: Optional[str]) -> Optional[str]:
        if out_path is None:
            return None
        pred_overlay = self._overlay_boundary(img, pred, color=(0, 255, 0))
        gt_overlay = self._overlay_boundary(img, gt, color=(0, 0, 255))
        comp = np.concatenate([pred_overlay, gt_overlay], axis=1)
        cv2.imwrite(out_path, comp)
        return out_path

    def _save_wavelet_bands(self, ll: np.ndarray, details: List[Tuple[np.ndarray, np.ndarray, np.ndarray]],
                            out_path: Optional[str]) -> Optional[str]:
        if out_path is None or details is None or len(details) == 0:
            return None
        lh, hl, hh = details[0]
        # Resize bands to same size
        h, w = ll.shape
        lh_r = cv2.resize(lh, (w, h), interpolation=cv2.INTER_LINEAR)
        hl_r = cv2.resize(hl, (w, h), interpolation=cv2.INTER_LINEAR)
        hh_r = cv2.resize(hh, (w, h), interpolation=cv2.INTER_LINEAR)
        top = np.concatenate([self._to_uint8(ll), self._to_uint8(lh_r)], axis=1)
        bottom = np.concatenate([self._to_uint8(hl_r), self._to_uint8(hh_r)], axis=1)
        mosaic = np.concatenate([top, bottom], axis=0)
        mosaic = cv2.cvtColor(mosaic, cv2.COLOR_GRAY2BGR)
        cv2.imwrite(out_path, mosaic)
        return out_path

    def _prepare_label_bundle(
        self,
        I_s: np.ndarray,
        M_s: np.ndarray,
        I_q: np.ndarray,
        case_id: Optional[str] = None,
        slice_id: Optional[int] = None,
        viz_dir: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
        gt_mask: Optional[np.ndarray] = None,
        verbose_log: bool = True,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        if logger is not None and verbose_log:
            log = logger.info
        elif logger is None and verbose_log:
            log = print
        else:
            log = lambda *args, **kwargs: None

        debug_dict: Dict[str, Any] = {'viz_paths': []}
        viz_task0 = os.path.join(viz_dir, 'overview') if viz_dir else None
        viz_task1 = os.path.join(viz_dir, 'task1') if viz_dir else None
        viz_task2 = os.path.join(viz_dir, 'task2') if viz_dir else None
        if viz_task0:
            self._ensure_dir(viz_task0)
        if viz_task1:
            self._ensure_dir(viz_task1)
        if viz_task2:
            self._ensure_dir(viz_task2)

        log('  [A1] Preprocess images...')
        Is_norm = self._preprocess_image(I_s)
        Iq_norm = self._preprocess_image(I_q)
        debug_dict['Is_norm'] = Is_norm
        debug_dict['Iq_norm'] = Iq_norm

        if Is_norm.shape[:2] != Iq_norm.shape[:2]:
            tgt_h, tgt_w = Iq_norm.shape[:2]
            Is_norm = cv2.resize(Is_norm.astype(np.float32), (tgt_w, tgt_h), interpolation=cv2.INTER_LINEAR)
            M_s = cv2.resize((M_s > 0.5).astype(np.uint8), (tgt_w, tgt_h), interpolation=cv2.INTER_NEAREST)

        log('  [A2] Compute support template...')
        M_s = (M_s > 0.5).astype(np.uint8)
        M_s = self._keep_largest_component(M_s)
        support_fill = binary_fill_holes(M_s > 0).astype(np.uint8)
        support_has_hole = support_fill.sum() > M_s.sum()
        if not support_has_hole:
            M_s = support_fill.astype(np.uint8)
        support_area = float(M_s.sum())
        support_desc = self._mask_descriptors(M_s)
        debug_dict['M_s_clean'] = M_s
        debug_dict['support_area'] = support_area
        debug_dict['support_desc'] = support_desc
        debug_dict['support_has_hole'] = bool(support_has_hole)

        log('  [A3] Compute query edge map...')
        Eq, Egrad = self._compute_edge_map(Iq_norm)
        debug_dict['Eq'] = Eq
        debug_dict['Egrad'] = Egrad

        log('  [A3.5] Compute DINO similarity...')
        Is_dino = Is_norm
        Iq_dino = Iq_norm
        if bool(self.config.enable_dino_freq_fusion):
            Is_dino, Iq_dino, dino_freq_debug = self._wavelet_mix_support_for_dino(Is_norm, Iq_norm)
            debug_dict['dino_freq_debug'] = dino_freq_debug
        Sdino, Fq_dino, Fs_dino, sim_maps, dino_region, dino_info = self._compute_dino_similarity(Is_dino, M_s, Iq_dino)
        debug_dict['Sdino'] = Sdino
        debug_dict['Fq_dino'] = Fq_dino
        debug_dict['Fs_dino'] = Fs_dino
        debug_dict['dino_sim_maps'] = sim_maps
        debug_dict['dino_region_mask'] = dino_region
        debug_dict['dino_region_info'] = dino_info

        log('  [A4] Generate pre-mask...')
        premask_out = self._generate_premask_chamfer(
            Iq_norm=Iq_norm,
            Is_norm=Is_norm,
            M_s=M_s,
            Sdino=Sdino,
            dino_region=dino_region,
            support_area=support_area,
            support_has_hole=bool(support_has_hole),
            support_desc=support_desc,
        )
        M_pre = premask_out['M_pre']
        debug_dict.update(premask_out.get('match_debug', {}))
        debug_dict['M_pre'] = M_pre

        log('  [A7] Compute query-native shape priors...')
        Dpre = self._compute_signed_distance(M_pre)
        Pin_pre, Pband_pre = self._compute_shape_priors(Dpre)
        debug_dict['Dpre'] = Dpre
        debug_dict['Pin_pre'] = Pin_pre
        debug_dict['Pband_pre'] = Pband_pre

        bundle = {
            'Is_norm': Is_norm,
            'Iq_norm': Iq_norm,
            'Eq': Eq,
            'Sdino': Sdino,
            'M_pre': M_pre,
            'Dpre': Dpre,
            'Pin_pre': Pin_pre,
            'Pband_pre': Pband_pre,
            'viz_task0': viz_task0,
            'viz_task1': viz_task1,
            'viz_task2': viz_task2,
        }
        return bundle, debug_dict

    def predict(
        self,
        support_image: np.ndarray,
        support_mask: np.ndarray,
        query_image: np.ndarray,
        *,
        case_id: Optional[str] = None,
        slice_id: Optional[int] = None,
        gt_mask: Optional[np.ndarray] = None,
        viz_dir: Optional[str] = None,
        logger: Optional[logging.Logger] = None,
    ) -> PredictionResult:
        bundle, debug = self._prepare_label_bundle(
            I_s=support_image,
            M_s=support_mask,
            I_q=query_image,
            case_id=case_id,
            slice_id=slice_id,
            viz_dir=viz_dir,
            logger=logger,
            gt_mask=gt_mask,
            verbose_log=bool(logger is not None),
        )
        M_pre = bundle['M_pre']
        P_pos, P_neg = self._sample_points_voronoi(
            M_pre,
            bundle['Pin_pre'],
            bundle['Pband_pre'],
            bundle['Sdino'],
            bundle['Dpre'],
            Iq=bundle['Iq_norm'],
            case_id=case_id,
            slice_id=slice_id,
            viz_dir=os.path.join(viz_dir, 'task6') if viz_dir else None,
        )
        if bool(self.config.enable_hmf):
            M_final, debug = self._run_hmf_branches(bundle=bundle, P_pos=P_pos, P_neg=P_neg, debug_dict=debug)
        else:
            M_final, debug = self._run_sam_with_points(
                bundle=bundle,
                P_pos=P_pos,
                P_neg=P_neg,
                case_id=case_id,
                slice_id=slice_id,
                logger=logger,
                gt_mask=gt_mask,
                verbose_log=bool(logger is not None),
                debug_dict=debug,
            )
        return PredictionResult(
            pred_mask=(M_final > 0.5).astype(np.uint8),
            pre_mask=(M_pre > 0.5).astype(np.uint8),
            points_pos=np.asarray(P_pos),
            points_neg=np.asarray(P_neg),
            bbox=debug.get('bbox'),
            debug_paths=[p for p in debug.get('viz_paths', []) if p],
            debug=debug,
        )
