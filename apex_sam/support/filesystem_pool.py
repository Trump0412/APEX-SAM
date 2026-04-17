from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from apex_sam.types import SupportMatch, SupportMeta


_SUPPORT_DIR_PATTERN = re.compile(r"case_(?P<case>.+?)_slice_(?P<slice>\d+)$")
_MASK_FILE_PATTERN = re.compile(r"mask_label(?P<label>\d+)\.npy$")


@dataclass
class _SupportEntry:
    support_id: str
    case_id: str
    slice_index: int
    image_path: Path
    mask_paths: dict[int, Path]
    descriptor: np.ndarray | None = None


class FileSystemSupportPool:
    """
    Public support provider used by the open-source release.

    Expected layout:
      support_pool/
        support_slices/                       # optional wrapper directory
          case_000_slice_015/
            image.npy
            mask_label1.npy
            mask_label2.npy
            meta.json                          # optional
    """

    def __init__(self, pool_dir: str, encoder: Any | None = None) -> None:
        self.pool_dir = Path(pool_dir).expanduser().resolve()
        self.encoder = encoder
        self.entries = self._scan_entries(self.pool_dir)
        self._image_cache: dict[str, np.ndarray] = {}
        self._descriptor_ready = False

    @staticmethod
    def _safe_int(value: Any, default: int = -1) -> int:
        try:
            return int(value)
        except Exception:
            return default

    def _parse_case_slice(self, support_dir: Path) -> tuple[str, int]:
        case_id = support_dir.name
        slice_index = -1
        match = _SUPPORT_DIR_PATTERN.match(support_dir.name)
        if match:
            case_id = str(match.group("case"))
            slice_index = int(match.group("slice"))
        meta_path = support_dir / "meta.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                case_id = str(meta.get("case_id", case_id))
                slice_index = self._safe_int(meta.get("slice_index"), default=slice_index)
            except Exception:
                pass
        return case_id, slice_index

    def _scan_entries(self, root: Path) -> list[_SupportEntry]:
        if not root.exists():
            raise RuntimeError(f"Support pool directory does not exist: {root}")
        support_root = root / "support_slices"
        if support_root.exists() and support_root.is_dir():
            scan_root = support_root
        else:
            scan_root = root

        entries: list[_SupportEntry] = []
        for support_dir in sorted([p for p in scan_root.rglob("*") if p.is_dir()]):
            image_path = support_dir / "image.npy"
            if not image_path.exists():
                continue
            mask_paths: dict[int, Path] = {}
            for mask_path in sorted(support_dir.glob("mask_label*.npy")):
                match = _MASK_FILE_PATTERN.match(mask_path.name)
                if match is None:
                    continue
                label_value = int(match.group("label"))
                mask_paths[label_value] = mask_path
            if not mask_paths:
                continue
            case_id, slice_index = self._parse_case_slice(support_dir)
            entries.append(
                _SupportEntry(
                    support_id=support_dir.name,
                    case_id=case_id,
                    slice_index=slice_index,
                    image_path=image_path,
                    mask_paths=mask_paths,
                )
            )
        if not entries:
            raise RuntimeError(
                f"No support slices found under {scan_root}. "
                "Expected folders that contain image.npy and mask_label*.npy."
            )
        return entries

    def _load_image(self, entry: _SupportEntry) -> np.ndarray:
        key = str(entry.image_path)
        if key not in self._image_cache:
            self._image_cache[key] = np.load(entry.image_path).astype(np.float32)
        return self._image_cache[key]

    @staticmethod
    def _l2_normalize(vec: np.ndarray) -> np.ndarray:
        vec = np.asarray(vec, dtype=np.float32).reshape(-1)
        return vec / (np.linalg.norm(vec) + 1e-8)

    def _compute_descriptor(self, image: np.ndarray) -> np.ndarray:
        if self.encoder is not None:
            return self._l2_normalize(self.encoder.compute_global_descriptor(image, None))
        stats = np.array(
            [
                float(np.mean(image)),
                float(np.std(image)),
                float(np.percentile(image, 25)),
                float(np.percentile(image, 75)),
            ],
            dtype=np.float32,
        )
        return self._l2_normalize(stats)

    def _ensure_descriptors(self) -> None:
        if self._descriptor_ready:
            return
        for entry in self.entries:
            entry.descriptor = self._compute_descriptor(self._load_image(entry))
        self._descriptor_ready = True

    def search(
        self,
        query_image: np.ndarray,
        *,
        label_value: int,
        topk: int,
        exclude_case_id: str | None = None,
        exclude_slice_index: int | None = None,
    ) -> list[SupportMatch]:
        self._ensure_descriptors()
        query_desc = self._compute_descriptor(np.asarray(query_image, dtype=np.float32))
        label_value = int(label_value)
        requested_topk = max(1, int(topk))

        candidates: list[tuple[float, _SupportEntry]] = []
        fallback_candidates: list[tuple[float, _SupportEntry]] = []

        for entry in self.entries:
            if label_value not in entry.mask_paths:
                continue
            if entry.descriptor is None:
                continue
            score = float(np.dot(entry.descriptor, query_desc))
            key = (entry.case_id, int(entry.slice_index))
            if (
                exclude_case_id is not None
                and exclude_slice_index is not None
                and key == (exclude_case_id, int(exclude_slice_index))
            ):
                fallback_candidates.append((score, entry))
                continue
            candidates.append((score, entry))

        if not candidates:
            candidates = fallback_candidates
        candidates.sort(key=lambda item: item[0], reverse=True)

        results: list[SupportMatch] = []
        for score, entry in candidates[:requested_topk]:
            image = self._load_image(entry)
            mask = np.load(entry.mask_paths[label_value]).astype(np.uint8)
            if mask.sum() <= 0:
                continue
            meta = SupportMeta(
                support_id=entry.support_id,
                case_id=entry.case_id,
                slice_path=str(entry.image_path),
                label_path=str(entry.mask_paths[label_value]),
                slice_index=int(entry.slice_index),
                label_value=label_value,
            )
            results.append(
                SupportMatch(
                    score=float(score),
                    meta=meta,
                    image=image,
                    mask=mask,
                )
            )
        return results
