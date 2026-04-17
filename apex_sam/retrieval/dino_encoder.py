from __future__ import annotations

import math
import os
import sys
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import KMeans


class DINOEncoder:
    def __init__(self, checkpoint: str, repo: str, model_name: str, device: str, p1: float = 1.0, p99: float = 99.0, dino_size: int = 512) -> None:
        self.checkpoint = checkpoint
        self.repo = repo
        self.model_name = model_name
        self.device = torch.device(device)
        self.p1 = p1
        self.p99 = p99
        self.dino_size = dino_size
        self.model = self._load_model()

    def _load_model(self):
        print(f"[DINOv3] Loading model: {self.checkpoint}")
        try:
            state = torch.load(self.checkpoint, map_location='cpu')
            if isinstance(state, dict) and 'model' in state:
                state = state['model']
            elif isinstance(state, dict) and 'state_dict' in state:
                state = state['state_dict']
            if any(k.startswith('module.') for k in state.keys()):
                state = {k.replace('module.', '', 1): v for k, v in state.items()}
            if os.path.isdir(self.repo):
                if self.repo not in sys.path:
                    sys.path.insert(0, self.repo)
                from dinov3.hub import backbones as d3_backbones
                model = getattr(d3_backbones, self.model_name)(pretrained=False)
            else:
                model = torch.hub.load(self.repo, self.model_name, pretrained=False, trust_repo=True)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing or unexpected:
                print(f"[DINOv3] load_state_dict: missing={len(missing)}, unexpected={len(unexpected)}")
            model = model.to(self.device).eval()
            print('[DINOv3] Model loaded')
            return model
        except Exception as exc:
            print(f'[DINOv3] Load failed: {exc}')
            print('[DINOv3] Using stub encoder')
            return None

    def preprocess(self, img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32)
        if img.ndim == 3:
            img = img.mean(axis=-1)
        p1, p99 = np.percentile(img, [self.p1, self.p99])
        img = np.clip(img, p1, p99)
        img = (img - img.min()) / (img.max() - img.min() + 1e-8)
        return img.astype(np.float32)

    def extract_features(self, img: np.ndarray) -> np.ndarray:
        img = self.preprocess(img)
        if self.model is None:
            h, w = img.shape
            return np.random.randn(h, w, 64).astype(np.float32)
        img_rgb = np.stack([img, img, img], axis=-1)
        img_resized = cv2.resize(img_rgb, (self.dino_size, self.dino_size))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_norm = (img_resized - mean) / std
        img_tensor = torch.from_numpy(img_norm).permute(2, 0, 1).unsqueeze(0).float().to(self.device)
        with torch.no_grad():
            output = self.model.forward_features(img_tensor)
            if isinstance(output, dict):
                tokens = output['x_norm_patchtokens']
            else:
                tokens = output[:, 1:, :]
            bsz, n_patches, channels = tokens.shape
            h_p = w_p = int(np.sqrt(n_patches))
            tokens = tokens.reshape(bsz, h_p, w_p, channels).permute(0, 3, 1, 2)
            features = F.interpolate(tokens, size=img.shape, mode='bilinear', align_corners=False)
            features = features.permute(0, 2, 3, 1).squeeze(0).cpu().numpy()
        return features.astype(np.float32)

    def compute_global_descriptor(self, img: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
        feats = self.extract_features(img)
        if mask is not None:
            mask_resized = cv2.resize(mask.astype(np.float32), (feats.shape[1], feats.shape[0]))
            mask_bool = mask_resized > 0.5
            if mask_bool.sum() > 0:
                desc = feats[mask_bool].mean(axis=0)
            else:
                desc = feats.mean(axis=(0, 1))
        else:
            desc = feats.mean(axis=(0, 1))
        desc = desc / (np.linalg.norm(desc) + 1e-8)
        return desc.astype(np.float32)


class DinoFeatureMixin:
    def _load_dinov3(self):
        self.dino_encoder = DINOEncoder(
            checkpoint=self.config.dinov3_checkpoint,
            repo=self.config.dinov3_repo,
            model_name=self.config.dinov3_model_name,
            device=self.config.device,
            p1=self.config.p1,
            p99=self.config.p99,
            dino_size=self.config.dino_size,
        )
        return self.dino_encoder.model

    def _extract_dino_features(self, img: np.ndarray) -> np.ndarray:
        return self.dino_encoder.extract_features(img)

    def _pad_to_multiple(self, arr: np.ndarray, multiple: int) -> Tuple[np.ndarray, Tuple[int, int]]:
        h, w = arr.shape
        pad_h = (multiple - h % multiple) % multiple
        pad_w = (multiple - w % multiple) % multiple
        if pad_h or pad_w:
            arr = np.pad(arr, ((0, pad_h), (0, pad_w)), mode="edge")
        return arr, (h, w)

    def _crop_to_shape(self, arr: np.ndarray, shape: Tuple[int, int]) -> np.ndarray:
        h, w = shape
        return arr[:h, :w]

    def _haar_dwt2(self, img: np.ndarray) -> Tuple[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        low_h = (img[:, 0::2] + img[:, 1::2]) * 0.5
        high_h = (img[:, 0::2] - img[:, 1::2]) * 0.5
        ll = (low_h[0::2, :] + low_h[1::2, :]) * 0.5
        lh = (low_h[0::2, :] - low_h[1::2, :]) * 0.5
        hl = (high_h[0::2, :] + high_h[1::2, :]) * 0.5
        hh = (high_h[0::2, :] - high_h[1::2, :]) * 0.5
        return ll, (lh, hl, hh)

    def _haar_idwt2(self, ll: np.ndarray, details: Tuple[np.ndarray, np.ndarray, np.ndarray]) -> np.ndarray:
        lh, hl, hh = details
        h, w = ll.shape
        low_h = np.zeros((h * 2, w), dtype=np.float32)
        high_h = np.zeros((h * 2, w), dtype=np.float32)
        low_h[0::2, :] = ll + lh
        low_h[1::2, :] = ll - lh
        high_h[0::2, :] = hl + hh
        high_h[1::2, :] = hl - hh
        img = np.zeros((h * 2, w * 2), dtype=np.float32)
        img[:, 0::2] = low_h + high_h
        img[:, 1::2] = low_h - high_h
        return img

    def _haar_dwt2_multi(self, img: np.ndarray, level: int):
        details = []
        cur = img
        for _ in range(level):
            cur, det = self._haar_dwt2(cur)
            details.append(det)
        return cur, details

    def _haar_idwt2_multi(self, ll: np.ndarray, details: List[Tuple[np.ndarray, np.ndarray, np.ndarray]]):
        cur = ll
        for det in reversed(details):
            cur = self._haar_idwt2(cur, det)
        return cur

    def _wavelet_mix_support_for_dino(
        self,
        Is_norm: np.ndarray,
        Iq_norm: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
        """DWT-based DINO fusion: replace support low-frequency with query low-frequency."""
        sup = Is_norm.astype(np.float32)
        qry = Iq_norm.astype(np.float32)
        if sup.ndim == 3:
            sup = sup.mean(axis=-1)
        if qry.ndim == 3:
            qry = qry.mean(axis=-1)

        sup = self._normalize(sup)
        qry = self._normalize(qry)

        if sup.shape != qry.shape:
            qry_to_sup = cv2.resize(qry, (sup.shape[1], sup.shape[0]), interpolation=cv2.INTER_LINEAR)
        else:
            qry_to_sup = qry

        level = int(self.config.dino_freq_level)
        min_dim = min(sup.shape)
        if min_dim < 2:
            level = 1
        else:
            max_level = int(math.floor(math.log2(min_dim)))
            level = max(1, min(level, max_level))

        multiple = 2 ** level
        sup_pad, sup_shape = self._pad_to_multiple(sup, multiple)
        qry_pad_sup, _ = self._pad_to_multiple(qry_to_sup, multiple)
        sup_ll, sup_details = self._haar_dwt2_multi(sup_pad, level)
        qry_ll, qry_details = self._haar_dwt2_multi(qry_pad_sup, level)
        # Keep support high-frequency, replace low-frequency with query.
        sup_mixed_pad = self._haar_idwt2_multi(qry_ll, sup_details)

        sup_mixed = self._normalize(self._crop_to_shape(sup_mixed_pad, sup_shape))
        qry_gray = qry.astype(np.float32)
        debug = {
            "support_ll": sup_ll,
            "support_details": sup_details,
            "query_ll": qry_ll,
            "query_details": qry_details,
            "support_mixed": sup_mixed,
            "query_gray": qry_gray,
        }
        return sup_mixed, qry_gray, debug

    def _compute_dino_similarity(
        self,
        Is_norm: np.ndarray,
        Ms: np.ndarray,
        Iq_norm: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray], Dict[str, Any]]:
        """Compute DINO similarity map and regional similarity masks."""
        # Extract features
        F_s = self._extract_dino_features(Is_norm)  # (H_s, W_s, C)
        F_q = self._extract_dino_features(Iq_norm)  # (H_q, W_q, C)

        # Normalize query features
        F_q_norm = F_q / (np.linalg.norm(F_q, axis=-1, keepdims=True) + 1e-8)

        # Prototype similarity (fallback)
        Ms_resized = cv2.resize(Ms.astype(np.float32), (F_s.shape[1], F_s.shape[0]))
        mask_bool = Ms_resized > 0.5
        if mask_bool.sum() > 0:
            prototype = F_s[mask_bool].mean(axis=0)  # (C,)
        else:
            prototype = F_s.mean(axis=(0, 1))  # (C,)
        prototype = prototype / (np.linalg.norm(prototype) + 1e-8)
        Sdino_proto = np.dot(F_q_norm, prototype)  # (H, W)
        Sdino_proto = (Sdino_proto + 1) / 2
        Sdino_proto = np.clip(Sdino_proto, 0, 1)

        # Regional similarity maps (K=24, top16)
        sim_maps = self._regional_similarity_maps(F_s, Ms, F_q_norm)
        if sim_maps is not None and sim_maps.size > 0:
            sim_mean = sim_maps.mean(axis=0)
            Sdino = self._normalize(sim_mean.astype(np.float32))
        else:
            sim_mean = None
            Sdino = Sdino_proto

        dino_region, dino_info = self._build_dino_region_mask(sim_maps, Sdino, sim_mean=sim_mean)
        return Sdino, F_q_norm, F_s, sim_maps, dino_region, dino_info

    def _regional_similarity_maps(
        self,
        Fs: np.ndarray,
        Ms: np.ndarray,
        Fq_norm: np.ndarray,
    ) -> np.ndarray:
        """Build regional similarity maps by KMeans over support-mask regions. Returns (R, Hq, Wq)."""
        if Fs is None or Fq_norm is None:
            return np.zeros((0, 1, 1), dtype=np.float32)
        Hs, Ws, C = Fs.shape
        Hq, Wq, _ = Fq_norm.shape
        mask = (Ms > 0).astype(np.uint8)
        if mask.shape != (Hs, Ws):
            mask = cv2.resize(mask.astype(np.uint8), (Ws, Hs), interpolation=cv2.INTER_NEAREST)
        coords = np.stack(np.nonzero(mask > 0), axis=1)
        if coords.shape[0] == 0:
            coords = np.array([[Hs // 2, Ws // 2]])

        n_regions = min(int(self.config.sim_num_regions), max(1, coords.shape[0]))
        try:
            kmeans = KMeans(n_clusters=n_regions, random_state=self.config.seed, n_init="auto")
        except TypeError:
            kmeans = KMeans(n_clusters=n_regions, random_state=self.config.seed, n_init=10)
        labels = kmeans.fit_predict(coords)

        prototypes = np.zeros((n_regions, C), dtype=np.float32)
        for r in range(n_regions):
            pts = coords[labels == r]
            if pts.size == 0:
                continue
            ys = pts[:, 0]
            xs = pts[:, 1]
            prototypes[r] = Fs[ys, xs].mean(axis=0)
        # normalize prototypes
        proto_norm = prototypes / (np.linalg.norm(prototypes, axis=1, keepdims=True) + 1e-8)

        # compute similarity maps
        Fq_flat = Fq_norm.reshape(-1, C)
        sim_flat = np.matmul(Fq_flat, proto_norm.T)  # (Hq*Wq, R)
        sim_maps = sim_flat.T.reshape(n_regions, Hq, Wq)
        sim_maps = sim_maps * float(self.config.sim_scaler)

        # keep top regions by max similarity
        region_scores = sim_maps.reshape(n_regions, -1).max(axis=1)
        order = np.argsort(region_scores)[::-1]
        keep = order[: min(int(self.config.sim_top_regions), len(order))]
        return sim_maps[keep]
