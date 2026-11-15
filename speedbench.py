# SpeedBench - Wall-clock timing of compression operations by matrix size
# Part of the Kapildhinwa2010-dev Toolkit.

"""SpeedBench - compression speed benchmark.

Lineage: speed-layer repos. Times four compression operations (SVD rank,
quantization, 1-D Fourier fit, hybrid) on random matrices of several
sizes using perf_counter, best-of-N repeats.

Timings are wall-clock on the machine running the benchmark and vary
with BLAS, CPU load and numpy build. Only relative comparisons within
one run are meaningful; no speedup claims are made here.
"""

from __future__ import annotations

import time

import numpy as np

__all__ = ["time_operation", "speed_table"]


def _fourier_basis(n_cols: int, num_coeffs: int) -> np.ndarray:
    t = np.linspace(0.0, 2.0 * np.pi, n_cols, endpoint=False)
    cols = [np.ones_like(t)]
    for k in range(1, num_coeffs + 1):
        cols.append(np.sin(k * t))
        cols.append(np.cos(k * t))
    return np.stack(cols, axis=1)


def _svd_compress(matrix: np.ndarray, rank: int = 4) -> np.ndarray:
    u, s, vt = np.linalg.svd(matrix, full_matrices=False)
    rank = min(rank, s.size)
    return (u[:, :rank] * s[:rank]) @ vt[:rank, :]


def _quantize(matrix: np.ndarray, bits: int = 8) -> np.ndarray:
    lo, hi = float(matrix.min()), float(matrix.max())
    levels = 2 ** bits - 1
    scale = (hi - lo) / levels if hi > lo else 1.0
    codes = np.clip(np.round((matrix - lo) / scale), 0, levels)
    return codes * scale + lo


def _fourier_compress(matrix: np.ndarray, num_coeffs: int = 4) -> np.ndarray:
    basis = _fourier_basis(matrix.shape[1], num_coeffs)
    coeffs, _, _, _ = np.linalg.lstsq(basis, matrix.T, rcond=None)
    return coeffs.T @ basis.T


def time_operation(func, repeats: int = 5) -> float:
    """Best-of-N wall-clock seconds for func(); best-of damps scheduler noise."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    best = None
    for _ in range(repeats):
        start = time.perf_counter()
        func()
        elapsed = time.perf_counter() - start
        if best is None or elapsed < best:
            best = elapsed
    return best


def speed_table(sizes=(64, 128, 256), repeats: int = 3, seed: int = 0) -> list:
    """Time each method per matrix size; returns rows of size/method/seconds."""
    if repeats < 1:
        raise ValueError("repeats must be >= 1")
    rng = np.random.default_rng(seed)
    rows = []
    for size in sizes:
        if size < 2:
            raise ValueError("sizes must be >= 2")
        matrix = rng.standard_normal((size, size))
        methods = {
            "svd_rank4": lambda: _svd_compress(matrix, rank=4),
            "quantize8": lambda: _quantize(matrix, bits=8),
            "fourier4": lambda: _fourier_compress(matrix, num_coeffs=4),
            "hybrid": lambda: _quantize(_svd_compress(matrix, rank=4), bits=8),
        }
        for name, func in methods.items():
            rows.append({
                "size": int(size),
                "method": name,
                "seconds": time_operation(func, repeats=repeats),
            })
    return rows
