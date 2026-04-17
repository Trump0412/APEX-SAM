import numpy as np

from apex_sam.evaluation.metrics import compute_dice, summarize_case_max_filtered


def test_dice_perfect_overlap():
    arr = np.ones((8, 8), dtype=np.float32)
    assert compute_dice(arr, arr) == 1.0


def test_dice_no_overlap():
    pred = np.ones((8, 8), dtype=np.float32)
    gt = np.zeros((8, 8), dtype=np.float32)
    assert compute_dice(pred, gt) == 0.0


def test_dice_both_empty():
    arr = np.zeros((8, 8), dtype=np.float32)
    assert compute_dice(arr, arr) == 1.0


def test_case_max_filtered_protocol():
    rows = [
        {"label": 1, "case_id": "000", "slice_id": 1, "dice": 0.2},
        {"label": 1, "case_id": "000", "slice_id": 2, "dice": 0.6},
        {"label": 1, "case_id": "001", "slice_id": 1, "dice": 0.05},
        {"label": 1, "case_id": "001", "slice_id": 3, "dice": 0.08},
        {"label": 1, "case_id": "002", "slice_id": 0, "dice": 0.9},
    ]
    out = summarize_case_max_filtered(rows, threshold=0.1)
    # Kept case maxima: 0.6 and 0.9
    assert abs(float(out["overall_mean_dice"]) - 0.75) < 1e-6
    assert int(out["num_case_entries"]) == 3
    assert int(out["num_case_entries_kept"]) == 2
