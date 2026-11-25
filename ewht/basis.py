from __future__ import annotations

import numpy as np

from ewht._ordering import degree_index, lex_index
from ewht._validate import validate_ps


def get_basis(
    bin_seqs: np.ndarray,
    ps: np.ndarray | None = None,
    *,
    basis: str = "eWHT",
    norm: str = "stable",
    degree_order: bool = True,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Build a Lasso design matrix from sampled binary genotypes.

    Each row evaluates the synthesis (inverse-transform) basis at one genotype so
    that ``y ≈ X @ coeffs`` uses the same coefficient convention as ``efwht``.

    Inputs:
    - ``bin_seqs``: ``(N, L)`` int8 array with ``0`` = wild-type and ``1`` = mutant.
    - ``ps``: ``(L,)`` float array of mutation probabilities. Ignored for ``WHT``
      (internally uses ``0.5``).
    - ``basis``: ``{'eWHT', 'WHT'}``.
    - ``norm``: ``{'stable', 'expectation', 'sqrt'}``. CS uses ``stable`` for eWHT.
    - ``degree_order``: if ``True``, columns are grouped by interaction degree.

    Outputs:
    - ``X``: ``(N, 2^L)`` float design matrix.
    - ``p_norms``: ``(L,)`` float array when ``norm='stable'`` and ``basis='eWHT'``;
      otherwise omitted (return is just ``X``).
    """
    basis = basis.upper()
    if basis == "WHT":
        bin_seqs = np.asarray(bin_seqs, dtype=np.int8)
        L = bin_seqs.shape[1]
        ps = validate_ps(None, L)
        norm = "expectation" if norm == "stable" else norm
    elif basis == "EWHT":
        pass
    else:
        raise ValueError("basis must be 'eWHT' or 'WHT'.")

    X, p_norms = _ewht_basis_matrix(bin_seqs, ps, norm=norm)
    if not degree_order:
        X = X[:, lex_index(bin_seqs.shape[1])]
    if p_norms is None:
        return X
    return X, p_norms


def ewht_basis(
    bin_seqs: np.ndarray,
    ps: np.ndarray,
    *,
    norm: str = "stable",
    degree_order: bool = True,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Alias for get_basis(..., basis='eWHT')."""
    return get_basis(bin_seqs, ps, basis="eWHT", norm=norm, degree_order=degree_order)


def wht_basis(
    bin_seqs: np.ndarray,
    *,
    norm: str = "expectation",
    degree_order: bool = True,
) -> np.ndarray:
    """Alias for get_basis(..., basis='WHT')."""
    out = get_basis(bin_seqs, None, basis="WHT", norm=norm, degree_order=degree_order)
    return out[0] if isinstance(out, tuple) else out


def restore_expectation_coeffs(
    coeffs_stable: np.ndarray,
    p_norms: np.ndarray,
    *,
    degree_order: bool = True,
) -> np.ndarray:
    """
    Convert stable-basis Lasso coefficients to expectation-basis coefficients.

    Inputs:
    - ``coeffs_stable``: ``(2^L,)`` float Lasso coefficients in degree order.
    - ``p_norms``: ``(L,)`` float stable scaling factors from ``get_basis``.
    - ``degree_order``: whether ``coeffs_stable`` is in degree order.

    Outputs:
    - ``(2^L,)`` float array of expectation-norm coefficients.
    """
    from ewht._ordering import reorder_coeffs
    from ewht.ewht import get_stable_scaling_factors

    coeffs_stable = np.asarray(coeffs_stable, dtype=float)
    p_norms = np.asarray(p_norms, dtype=float)
    L = len(p_norms)
    scaling_lex = get_stable_scaling_factors(p_norms)
    if degree_order:
        scaling = reorder_coeffs(scaling_lex, L, degree_order=True)
    else:
        scaling = scaling_lex
    return coeffs_stable / scaling


def _genotype_indices(bin_seqs: np.ndarray) -> np.ndarray:
    L = bin_seqs.shape[1]
    powers = 1 << np.arange(L - 1, -1, -1, dtype=np.int64)
    return bin_seqs.astype(np.int64) @ powers


def _p_bias_matrix(p: float, *, stable: bool = False) -> tuple[np.ndarray, float]:
    q = 1.0 - p
    c0, c1 = np.sqrt(p / q), -np.sqrt(q / p)
    p_norm = 1.0
    if stable:
        p_norm = np.sqrt(c0**2 + c1**2)
        c0 /= p_norm
        c1 /= p_norm
    return np.array([[1.0, 1.0], [c0, c1]], dtype=float), p_norm


def _synthesis_matrix(
    ps: np.ndarray,
    *,
    norm: str = "expectation",
) -> tuple[np.ndarray, np.ndarray | None]:
    """Full (2^L x 2^L) synthesis matrix with lexicographic coefficient columns."""
    is_stable = norm == "stable"
    ps = np.asarray(ps, dtype=float)
    n = len(ps)
    p_norms = np.ones(n, dtype=np.float64)

    H, pn = _p_bias_matrix(float(ps[0]), stable=is_stable)
    p_norms[0] = pn
    for i in range(1, n):
        mat, pn = _p_bias_matrix(float(ps[i]), stable=is_stable)
        p_norms[i] = pn
        H = np.kron(H, mat)

    S = H.T
    if norm == "sqrt":
        S = S / np.sqrt(1 << n)
        return S, p_norms if is_stable else None
    if norm in ("expectation", "stable"):
        return S, p_norms if is_stable else None
    raise ValueError(f"norm must be one of {{'expectation', 'sqrt', 'stable'}}, got {norm!r}.")


def _ewht_basis_matrix(
    bin_seqs: np.ndarray,
    ps: np.ndarray,
    *,
    norm: str = "stable",
) -> tuple[np.ndarray, np.ndarray | None]:
    bin_seqs = np.asarray(bin_seqs, dtype=np.int8)
    ps = np.asarray(ps, dtype=np.float64)
    N, L = bin_seqs.shape

    S, p_norms = _synthesis_matrix(ps, norm=norm)
    row_idx = _genotype_indices(bin_seqs)
    X = S[row_idx, :]
    X = X[:, degree_index(L)]
    return X, p_norms
