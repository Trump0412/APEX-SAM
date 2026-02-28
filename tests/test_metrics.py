import numpy as np

from apex_sam.evaluation.metrics import compute_dice


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
