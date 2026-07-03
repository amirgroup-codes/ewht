"""Evolutionary Walsh-Hadamard transform (eWHT) for protein fitness landscapes."""

from __future__ import annotations

import numpy as np

try:
    import torch
except ImportError:
    torch = None

from ewht._validate import validate_transform_inputs


def _norm_multiplier(N: int, norm: str | None) -> float:
    if norm in ("expectation", "stable"):
        return 1.0
    if norm == "sqrt":
        return float(np.sqrt(N))
    if norm is None:
        return float(N)
    raise ValueError("norm must be one of {'expectation', 'sqrt', 'stable', None}.")


def efwht(
    y: np.ndarray,
    ps: np.ndarray | None = None,
    norm: str | None = "expectation",
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Forward evolutionary Walsh-Hadamard transform in O(N log N) time.

    Computes eWHT coefficients of a fitness landscape under per-site mutation
    probabilities ``ps``. Coefficients are in lexicographic genotype order unless
    you reorder them explicitly.

    Inputs:
    - ``y``: ``(N,)`` float array with ``N = 2^L`` landscape fitness values in
      lexicographic genotype order (bit 0 = wild-type, bit 1 = mutant).
    - ``ps``: ``(L,)`` float array of per-site mutation probabilities
      ``p(mutant)``. If ``None``, defaults to ``0.5`` at every site (standard WHT).
    - ``norm``: ``{'expectation', 'sqrt', 'stable', None}``. Use the same value
      for ``iefwht`` to round-trip exactly.

    Outputs:
    - ``(N,)`` float array of eWHT coefficients when ``norm != 'stable'``.
    - ``(coeffs, p_norms)`` tuple when ``norm == 'stable'``, where ``coeffs`` is
      ``(N,)`` float and ``p_norms`` is ``(L,)`` float stable scaling factors.
    """
    return _compute_efwht(y, ps=ps, add_bias=True, norm=norm)


def iefwht(
    coeffs: np.ndarray | tuple[np.ndarray, np.ndarray],
    ps: np.ndarray | None = None,
    norm: str | None = "expectation",
    *,
    p_norms: np.ndarray | None = None,
) -> np.ndarray:
    """
    Inverse evolutionary Walsh-Hadamard transform.

    Reconstructs a landscape from eWHT coefficients. Pass the same ``norm`` used
    in the forward ``efwht`` call for an exact round-trip.

    Inputs:
    - ``coeffs``: ``(N,)`` float array of coefficients, or ``(coeffs, p_norms)``
      tuple returned by ``efwht(..., norm='stable')``.
    - ``ps``: ``(L,)`` float array of per-site mutation probabilities (same as forward).
    - ``norm``: ``{'expectation', 'sqrt', 'stable', None}``.
    - ``p_norms``: ``(L,)`` float array required when ``norm='stable'`` and
      ``coeffs`` is not passed as a tuple.

    Outputs:
    - ``(N,)`` float array of reconstructed landscape values in lexicographic order.
    """
    if norm == "stable":
        if isinstance(coeffs, tuple):
            coeffs, p_norms = coeffs
        if p_norms is None:
            raise ValueError(
                "norm='stable' requires p_norms from efwht, e.g. "
                "iefwht((coeffs, p_norms), ps, norm='stable')."
            )
        coeffs = stable_to_expectation_coeffs(coeffs, p_norms, degree_order=False)
        norm = "expectation"

    if isinstance(coeffs, tuple):
        coeffs = coeffs[0]

    if norm not in ("expectation", "sqrt", None):
        raise ValueError("norm must be one of {'expectation', 'sqrt', 'stable', None}.")

    yv, p, N, _ = validate_transform_inputs(np.asarray(coeffs, dtype=float), ps)
    y = _synthesis_butterfly_numpy(yv, p)
    return y / _inverse_output_scale(N, norm)


def _inverse_output_scale(N: int, norm: str | None) -> float:
    if norm in ("expectation", "stable"):
        return 1.0
    if norm == "sqrt":
        return float(np.sqrt(N))
    if norm is None:
        return float(N)
    raise ValueError("norm must be one of {'expectation', 'sqrt', 'stable', None}.")


def stable_to_expectation_coeffs(
    coeffs_stable: np.ndarray,
    p_norms: np.ndarray,
    *,
    degree_order: bool = False,
) -> np.ndarray:
    """
    Convert stable-norm eWHT coefficients to expectation-norm coefficients.

    Inputs:
    - ``coeffs_stable``: ``(N,)`` float array from ``efwht(..., norm='stable')``.
    - ``p_norms``: ``(L,)`` float stable scaling factors from the forward transform.
    - ``degree_order``: if ``True``, ``coeffs_stable`` is in degree order; otherwise
      lexicographic order.

    Outputs:
    - ``(N,)`` float array of expectation-norm coefficients.
    """
    from ewht._ordering import reorder_coeffs

    coeffs_stable = np.asarray(coeffs_stable, dtype=float)
    p_norms = np.asarray(p_norms, dtype=float)
    L = len(p_norms)
    scaling_lex = get_stable_scaling_factors(p_norms)
    if degree_order:
        scaling = reorder_coeffs(scaling_lex, L, degree_order=True)
        return coeffs_stable * scaling
    return coeffs_stable * scaling_lex


def get_stable_scaling_factors(p_norms: np.ndarray) -> np.ndarray:
    """
    Kronecker scaling factors mapping stable-norm to expectation-norm coefficients.

    Inputs:
    - ``p_norms``: ``(L,)`` float array from ``efwht(..., norm='stable')``.

    Outputs:
    - ``(N,)`` float array in lexicographic coefficient order with ``N = 2^L``.
    """
    p_norms = np.asarray(p_norms, dtype=float)
    if torch is not None and torch.is_tensor(p_norms):
        scaling = torch.ones(1, dtype=torch.float64, device=p_norms.device)
        for i in range(len(p_norms) - 1, -1, -1):
            scaling = torch.cat([scaling, scaling * p_norms[i]])
        return scaling.cpu().numpy()
    scaling = np.array([1.0], dtype=np.float64)
    for pn in reversed(p_norms):
        scaling = np.concatenate([scaling, scaling * pn])
    return scaling


def _compute_efwht(
    y: np.ndarray,
    ps: np.ndarray | None = None,
    *,
    add_bias: bool = True,
    norm: str | None = "expectation",
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    input_is_torch = torch is not None and torch.is_tensor(y)
    if input_is_torch:
        return _compute_efwht_torch(
            y, ps, add_bias=add_bias, norm=norm, device=y.device, return_numpy=False
        )
    yv, p, _, _ = validate_transform_inputs(y, ps)
    if len(p) > 14 and torch is not None and torch.cuda.is_available():
        return _compute_efwht_torch(yv, p, add_bias=add_bias, norm=norm, device="cuda", return_numpy=True)
    return _compute_efwht_numpy(yv, p, add_bias=add_bias, norm=norm)


def _compute_efwht_numpy(
    y: np.ndarray,
    ps: np.ndarray,
    *,
    add_bias: bool = False,
    norm: str | None = "expectation",
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    y = y.copy()
    n = len(ps)
    N = len(y)
    p_norms = np.ones(n, dtype=np.float64)

    if add_bias:
        norm_scale = _norm_multiplier(N, norm)
        idxs = np.arange(N)
        binaries = (idxs[:, None] & (1 << np.arange(n - 1, -1, -1))) > 0
        bias_factors = np.prod(np.where(binaries, ps, 1 - ps), axis=1) * norm_scale
        y *= bias_factors

    h = 1
    while h < N:
        level_idx = n - 1 - int(np.log2(h))
        p = ps[level_idx]
        alpha, beta = np.sqrt(p / (1 - p)), np.sqrt((1 - p) / p)
        if norm == "stable":
            pn = np.sqrt(alpha**2 + beta**2)
            p_norms[level_idx] = pn
            alpha /= pn
            beta /= pn
        y = y.reshape(-1, 2, h)
        x, a = y[:, 0, :].copy(), y[:, 1, :].copy()
        y[:, 0, :] = x + a
        y[:, 1, :] = (alpha * x) - (beta * a)
        h *= 2

    res = y.flatten()
    return (res, p_norms) if norm == "stable" else res


def _synthesis_butterfly_numpy(a: np.ndarray, ps: np.ndarray) -> np.ndarray:
    """Expectation-norm synthesis butterfly (inverse of the analysis butterfly)."""
    h = 1
    while h < len(a):
        p = ps[-int(np.log2(h)) - 1]
        alpha = np.sqrt(p / (1 - p))
        beta = np.sqrt((1 - p) / p)
        for i in range(0, len(a), h * 2):
            for j in range(i, i + h):
                x = a[j]
                y = a[j + h]
                a[j] = x + (alpha * y)
                a[j + h] = x - (beta * y)
        h *= 2
    return a


def _compute_efwht_torch(
    y: np.ndarray,
    ps: np.ndarray,
    *,
    add_bias: bool = False,
    norm: str | None = "expectation",
    device: str = "cuda",
    return_numpy: bool = False,
):
    if torch is None:
        raise ImportError("torch is required for the GPU implementation.")
    if isinstance(y, np.ndarray):
        y_t = torch.as_tensor(y, dtype=torch.float64, device=device)
    else:
        y_t = y.to(dtype=torch.float64, device=device)
    if not return_numpy:
        y_t = y_t.clone()

    ps_t = torch.as_tensor(ps, dtype=torch.float64, device=device)
    n = len(ps_t)
    N = len(y_t)
    p_norms_t = torch.ones(n, dtype=torch.float64, device=device)

    if add_bias:
        norm_scale = _norm_multiplier(N, norm)
        idxs = torch.arange(N, device=device)
        binaries = (idxs[:, None] & (1 << torch.arange(n - 1, -1, -1, device=device))) > 0
        bias_factors = torch.prod(torch.where(binaries, ps_t, 1 - ps_t), dim=1) * norm_scale
        y_t *= bias_factors

    h = 1
    while h < N:
        level_idx = n - 1 - int(np.log2(h))
        p = ps_t[level_idx]
        alpha, beta = torch.sqrt(p / (1 - p)), torch.sqrt((1 - p) / p)
        if norm == "stable":
            pn = torch.sqrt(alpha**2 + beta**2)
            p_norms_t[level_idx] = pn
            alpha /= pn
            beta /= pn
        y_t = y_t.view(-1, 2, h)
        x, a = y_t[:, 0, :].clone(), y_t[:, 1, :].clone()
        y_t[:, 0, :] = x + a
        y_t[:, 1, :] = (alpha * x) - (beta * a)
        y_t = y_t.view(-1)
        h *= 2

    if norm == "stable":
        return (
            (y_t.cpu().numpy(), p_norms_t.cpu().numpy())
            if return_numpy
            else (y_t, p_norms_t)
        )
    return y_t.cpu().numpy() if return_numpy else y_t


def efwht_from_dataframe(
    data,
    ps: np.ndarray,
    *,
    basis: str = "eWHT",
    norm: str | None = "expectation",
    center_by_ps: bool = True,
) -> tuple[np.ndarray, float]:
    """
    Forward eWHT from a preprocessed DataFrame or ``(bin_seqs, y)`` tuple.

    Parses genotype and fitness columns, optionally centers targets the same way
    as ``run_cs_experiment``, and runs ``efwht``.

    Inputs:
    - ``data``: pandas DataFrame with ``genotype`` and fitness columns, or
      ``(bin_seqs, y)`` with ``bin_seqs`` ``(N, L)`` int8 and ``y`` ``(N,)`` float.
    - ``ps``: ``(L,)`` float mutation probabilities.
    - ``basis``: ``{'eWHT', 'WHT'}`` centering convention when ``center_by_ps=True``.
    - ``norm``: passed through to ``efwht``.
    - ``center_by_ps``: if ``True``, subtract ps-weighted (eWHT) or uniform (WHT) mean.

    Outputs:
    - ``coeffs``: ``(N,)`` float eWHT coefficients with ``N = len(y)`` (typically ``2^L``).
    - ``center``: float mean subtracted before the transform (``0.0`` if not centered).
    """
    from ewht._io import parse_seq_data
    from ewht.cs import center_targets

    bin_seqs, y = parse_seq_data(data)
    if center_by_ps:
        y, center = center_targets(y, bin_seqs, ps, basis=basis)
    else:
        center = 0.0
    coeffs = efwht(y, ps=ps, norm=norm)
    if isinstance(coeffs, tuple):
        coeffs = coeffs[0]
    return np.asarray(coeffs, dtype=float), center
