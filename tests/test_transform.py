"""Transform round-trips, normalizations and the design-matrix / coefficient contract."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import ewht


def genotypes(L: int) -> np.ndarray:
    """All 2^L genotypes in lexicographic order (site 0 = most significant bit)."""
    return np.array(list(itertools.product((0, 1), repeat=L)), dtype=np.int8)


def degree_order(L: int) -> list[int]:
    """Site i is bit 1 << (L - 1 - i), matching the transform's convention."""
    return [
        sum(1 << (L - 1 - i) for i in combo)
        for r in range(L + 1)
        for combo in itertools.combinations(range(L), r)
    ]


def test_degree_slot_holds_the_interaction_it_is_labelled_with():
    """A landscape driven by one site must put its effect in that site's slot."""
    L = 4
    ps = np.array([0.3, 0.4, 0.6, 0.7])
    combos = [
        combo for r in range(L + 1) for combo in itertools.combinations(range(L), r)
    ]
    for site in range(L):
        g = genotypes(L)
        y = g[:, site].astype(float)
        X = ewht.get_basis(g, ps, basis="eWHT", norm="expectation", degree_order=True)
        nonzero = [
            k for k in np.flatnonzero(np.abs(np.linalg.solve(X, y)) > 1e-9) if k != 0
        ]
        assert [combos[k] for k in nonzero] == [(site,)]


@pytest.mark.parametrize("L", [1, 3, 6])
@pytest.mark.parametrize("norm", ["expectation", "sqrt", None])
def test_roundtrip(L, norm):
    rng = np.random.default_rng(L)
    ps = rng.uniform(0.1, 0.9, L)
    y = rng.normal(size=2**L)
    coeffs = ewht.efwht(y, ps, norm=norm)
    assert np.allclose(ewht.iefwht(coeffs, ps, norm=norm), y)


def test_roundtrip_stable():
    rng = np.random.default_rng(0)
    ps = rng.uniform(0.1, 0.9, 5)
    y = rng.normal(size=2**5)
    coeffs, p_norms = ewht.efwht(y, ps, norm="stable")
    assert np.allclose(ewht.iefwht((coeffs, p_norms), ps, norm="stable"), y)
    assert np.allclose(
        ewht.stable_to_expectation_coeffs(coeffs, p_norms), ewht.efwht(y, ps)
    )


def test_inputs_are_not_modified():
    rng = np.random.default_rng(1)
    ps = rng.uniform(0.1, 0.9, 4)
    y = rng.normal(size=2**4)
    y_before = y.copy()
    coeffs = ewht.efwht(y, ps)
    coeffs_before = coeffs.copy()
    ewht.iefwht(coeffs, ps)
    assert np.allclose(y, y_before)
    assert np.allclose(coeffs, coeffs_before)


def test_uniform_ps_is_the_plain_wht():
    rng = np.random.default_rng(2)
    y = rng.normal(size=2**4)
    assert np.allclose(ewht.efwht(y, np.full(4, 0.5)), ewht.efwht(y, None))


def test_expectation_norm_is_the_variance_under_mu_p():
    """sum of squared coefficients (bar the constant) = Var_{mu_p}(y)."""
    rng = np.random.default_rng(3)
    L = 5
    ps = rng.uniform(0.15, 0.85, L)
    y = rng.normal(size=2**L)
    weights = np.array([1.0])
    for p in ps:
        weights = np.kron(weights, [1 - p, p])
    mean = float(np.sum(weights * y))
    variance = float(np.sum(weights * (y - mean) ** 2))
    coeffs = ewht.efwht(y - mean, ps)
    assert np.isclose(float(np.sum(coeffs[1:] ** 2)), variance)


@pytest.mark.parametrize("basis", ["eWHT", "WHT"])
def test_design_matrix_matches_coefficients(basis):
    """y = X @ coeffs: the basis and the transform must use one convention."""
    rng = np.random.default_rng(4)
    L = 4
    ps = rng.uniform(0.15, 0.85, L)
    y = rng.normal(size=2**L)
    transform_ps = ps if basis == "eWHT" else np.full(L, 0.5)
    coeffs = ewht.efwht(y, transform_ps)[degree_order(L)]
    X = ewht.get_basis(genotypes(L), ps, basis=basis, norm="expectation", degree_order=True)
    assert np.allclose(X @ coeffs, y)


def test_stable_basis_restores_expectation_coefficients():
    rng = np.random.default_rng(5)
    L = 4
    ps = rng.uniform(0.15, 0.85, L)
    y = rng.normal(size=2**L)
    expected = ewht.efwht(y, ps)[degree_order(L)]
    X, p_norms = ewht.get_basis(genotypes(L), ps, basis="eWHT", norm="stable")
    coeffs_stable = np.linalg.solve(X, y)
    assert np.allclose(ewht.restore_expectation_coeffs(coeffs_stable, p_norms), expected)


def test_rejects_bad_input():
    with pytest.raises(ValueError):
        ewht.efwht(np.ones(6))  # not a power of two
    with pytest.raises(ValueError):
        ewht.efwht(np.ones(8), np.array([0.5, 0.5]))  # wrong ps length
    with pytest.raises(ValueError):
        ewht.efwht(np.ones(8), np.array([0.0, 0.5, 0.5]))  # p outside (0, 1)
