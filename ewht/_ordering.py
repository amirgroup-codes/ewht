from __future__ import annotations

from itertools import combinations

import numpy as np


def degree_index(L: int) -> np.ndarray:
    """Map degree-ordered coefficient index -> lexicographic genotype index."""
    idx = [
        sum(1 << i for i in combo)
        for r in range(L + 1)
        for combo in combinations(range(L), r)
    ]
    return np.asarray(idx, dtype=np.int64)


def lex_index(L: int) -> np.ndarray:
    """Map lexicographic index -> degree-ordered index."""
    deg = degree_index(L)
    lex = np.empty_like(deg)
    lex[deg] = np.arange(len(deg))
    return lex


def reorder_coeffs(coeffs: np.ndarray, L: int, *, degree_order: bool = False) -> np.ndarray:
    """Convert coefficients between lexicographic and degree orderings."""
    coeffs = np.asarray(coeffs, dtype=float)
    if degree_order:
        return coeffs[degree_index(L)]
    return coeffs[lex_index(L)]
