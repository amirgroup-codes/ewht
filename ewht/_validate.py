from __future__ import annotations

import numpy as np


def validate_ps(ps: np.ndarray | None, L: int) -> np.ndarray:
    if ps is None:
        return np.full(L, 0.5, dtype=float)
    p = np.asarray(ps, dtype=float)
    if p.shape != (L,):
        raise ValueError(f"ps must have shape ({L},).")
    if np.any((p <= 0.0) | (p >= 1.0)):
        raise ValueError("All ps entries must lie strictly between 0 and 1.")
    return p


def validate_transform_inputs(y: np.ndarray, ps: np.ndarray | None) -> tuple[np.ndarray, np.ndarray, int, int]:
    a = np.asarray(y, dtype=float)
    if a.ndim != 1:
        raise ValueError("y must be a 1D array.")
    N = len(a)
    if N == 0 or (N & (N - 1)) != 0:
        raise ValueError("Input length must be a power of 2.")
    L = int(np.log2(N))
    p = validate_ps(ps, L)
    return a.copy(), p, N, L
