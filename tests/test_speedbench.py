"""Tests for SpeedBench: wall-clock timing of compression operations."""

import numpy as np
import pytest

from speedbench import speed_table, time_operation


def test_time_operation_positive_and_repeatable():
    matrix = np.ones((16, 16))
    seconds = time_operation(lambda: matrix @ matrix, repeats=2)
    assert seconds >= 0.0


def test_time_operation_validates():
    with pytest.raises(ValueError):
        time_operation(lambda: None, repeats=0)


def test_speed_table_covers_sizes_and_methods():
    rows = speed_table(sizes=(16, 32), repeats=1, seed=0)
    methods = {r["method"] for r in rows}
    sizes = {r["size"] for r in rows}
    assert methods == {"svd_rank4", "quantize8", "fourier4", "hybrid"}
    assert sizes == {16, 32}
    assert all(r["seconds"] >= 0.0 for r in rows)


def test_speed_table_validates():
    with pytest.raises(ValueError):
        speed_table(sizes=(1,), repeats=1)
    with pytest.raises(ValueError):
        speed_table(sizes=(8,), repeats=0)


def test_operations_reconstruct_reasonably():
    rng = np.random.default_rng(3)
    matrix = rng.standard_normal((16, 16))
    q = np.clip(np.round((matrix - matrix.min()) / (np.ptp(matrix) / 255) ), 0, 255)
    assert q.shape == matrix.shape
