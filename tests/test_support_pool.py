from pathlib import Path

import numpy as np

from apex_sam.support.filesystem_pool import FileSystemSupportPool


def test_support_pool_search(tmp_path: Path):
    support_dir = tmp_path / "support_slices" / "case_000_slice_005"
    support_dir.mkdir(parents=True, exist_ok=True)
    image = np.ones((16, 16), dtype=np.float32)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:12, 4:12] = 1
    np.save(support_dir / "image.npy", image)
    np.save(support_dir / "mask_label1.npy", mask)

    pool = FileSystemSupportPool(str(tmp_path))
    matches = pool.search(np.ones((16, 16), dtype=np.float32), label_value=1, topk=1)
    assert len(matches) == 1
    assert matches[0].image is not None
    assert matches[0].mask is not None
    assert matches[0].meta.case_id == "000"
    assert int(matches[0].meta.slice_index) == 5
