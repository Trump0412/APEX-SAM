from pathlib import Path

import numpy as np

from apex_sam.retrieval.local_db import LocalSupportDB


class DummyEncoder:
    def compute_global_descriptor(self, image, mask=None):
        image = np.asarray(image, dtype=np.float32)
        val = float(image.mean())
        return np.array([val, val + 1.0], dtype=np.float32)


def test_local_db_roundtrip(tmp_path: Path):
    path = tmp_path / 'db.npz'
    descriptors = np.array([[1.0, 2.0], [2.0, 3.0]], dtype=np.float32)
    meta = np.array([
        {'slice_path': 'a', 'label_path': 'la', 'slice_index': 0},
        {'slice_path': 'b', 'label_path': 'lb', 'slice_index': 1},
    ], dtype=object)
    np.savez_compressed(path, descriptors=descriptors, meta=meta)
    db = LocalSupportDB.load(str(path), encoder=DummyEncoder())
    results = db.search(np.ones((4, 4), dtype=np.float32), topk=1)
    assert len(results) == 1
    assert results[0].meta.slice_path in {'a', 'b'}
