from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from apex_sam.data.chaos_mr import iter_cases, load_case
from apex_sam.types import SupportMatch, SupportMeta


@dataclass
class LocalSupportDB:
    descriptors: np.ndarray
    meta: list[dict[str, Any]]
    encoder: Any | None = None

    @classmethod
    def build(cls, data_dir: str, output_path: str, encoder: Any) -> str:
        descriptors: list[np.ndarray] = []
        meta: list[dict[str, Any]] = []
        for image_path, label_path in iter_cases(data_dir):
            image_volume, _ = load_case(image_path, label_path)
            for slice_index in range(image_volume.shape[0]):
                desc = encoder.compute_global_descriptor(image_volume[slice_index], None)
                descriptors.append(desc.astype(np.float32))
                meta.append({
                    'slice_path': image_path,
                    'label_path': label_path,
                    'slice_index': int(slice_index),
                })
        if not descriptors:
            raise RuntimeError('Local DB builder found no slices.')
        descriptors_np = np.stack(descriptors, axis=0)
        output = Path(output_path)
        output.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(output, descriptors=descriptors_np, meta=np.array(meta, dtype=object))
        return str(output)

    @classmethod
    def load(cls, path: str, encoder: Any | None = None) -> 'LocalSupportDB':
        data = np.load(path, allow_pickle=True)
        descriptors = data['descriptors'].astype(np.float32)
        meta = list(data['meta'])
        return cls(descriptors=descriptors, meta=meta, encoder=encoder)

    def attach_encoder(self, encoder: Any) -> 'LocalSupportDB':
        self.encoder = encoder
        return self

    def search(self, query_image: np.ndarray, *, topk: int = 5) -> list[SupportMatch]:
        if self.encoder is None:
            raise RuntimeError('LocalSupportDB.search requires an attached encoder.')
        q_desc = self.encoder.compute_global_descriptor(query_image, None)
        sims = self.descriptors @ q_desc
        order = np.argsort(sims)[::-1]
        results: list[SupportMatch] = []
        for idx in order[: max(1, int(topk))]:
            item = self.meta[int(idx)]
            if isinstance(item, np.ndarray):
                item = item.item()
            meta = SupportMeta(
                slice_path=item.get('slice_path', ''),
                label_path=item.get('label_path', ''),
                slice_index=int(item.get('slice_index', 0)),
            )
            results.append(SupportMatch(score=float(sims[int(idx)]), meta=meta))
        return results
