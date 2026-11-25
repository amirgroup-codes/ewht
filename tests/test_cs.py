"""Compressed sensing: the paper's split, lambda rule and recovery behaviour."""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

import ewht


def sparse_landscape(L: int, n_nonzero: int, seed: int = 0):
    """A landscape with a few low-order eWHT coefficients, and its genotype table."""
    rng = np.random.default_rng(seed)
    ps = rng.uniform(0.2, 0.8, L)
    orders = np.array([bin(i).count("1") for i in range(2**L)])
    candidates = np.flatnonzero((orders > 0) & (orders <= 2))
    coeffs = np.zeros(2**L)
    coeffs[rng.choice(candidates, size=n_nonzero, replace=False)] = rng.normal(
        size=n_nonzero
    )
    y = ewht.iefwht(coeffs, ps)
    genotypes = ["".join(map(str, g)) for g in itertools.product((0, 1), repeat=L)]
    return ps, coeffs, pd.DataFrame({"genotype": genotypes, "fitness": y})


def test_recovers_a_sparse_landscape_from_few_measurements():
    L = 8
    ps, coeffs, df = sparse_landscape(L, n_nonzero=6)
    train, test = ewht.split_bin_train_test(df, 120, pool_size=120, random_state=0)
    result = ewht.run_cs_experiment(train, test, ps, basis="eWHT")
    assert result.test_metrics["r2"] > 0.9
    recovered = set(np.flatnonzero(np.abs(result.coeffs) > 0.05).tolist())
    strong = set(np.flatnonzero(np.abs(coeffs) > 0.3).tolist())
    assert strong and strong.issubset(recovered)


def test_coefficients_come_back_in_the_transform_convention():
    """A fit on the whole library must land on the eWHT coefficients themselves."""
    L = 5
    ps, coeffs, df = sparse_landscape(L, n_nonzero=4, seed=1)
    full = (
        np.stack([np.fromiter(g, dtype=np.int8) for g in df["genotype"]]),
        df["fitness"].to_numpy(float),
    )
    result = ewht.run_cs_experiment(
        full, full, ps, basis="eWHT", lambda_selection="grid", lambdas=[1e-8]
    )
    assert np.allclose(result.coeffs[1:], coeffs[1:], atol=1e-3)
    assert result.test_metrics["r2"] > 0.999


def test_lambda_path_and_grid_both_run():
    L = 6
    ps, _, df = sparse_landscape(L, n_nonzero=5, seed=2)
    train, test = ewht.split_bin_train_test(df, 40, pool_size=40, random_state=0)
    path = ewht.run_cs_experiment(train, test, ps, lambda_selection="path")
    grid = ewht.run_cs_experiment(train, test, ps, lambda_selection="grid")
    assert path.best_lambda > 0 and grid.best_lambda > 0
    assert len(path.cv_scores) == ewht.cs.LAMBDA_PATH_SIZE
    assert set(grid.cv_scores) == set(ewht.cs.DEFAULT_LAMBDAS)


def test_grid_lambdas_are_rejected_for_the_path_rule():
    ps, _, df = sparse_landscape(4, n_nonzero=2, seed=3)
    train, test = ewht.split_bin_train_test(df, 10, pool_size=10, random_state=0)
    with pytest.raises(ValueError):
        ewht.run_cs_experiment(train, test, ps, lambda_selection="path", lambdas=[0.1])


def test_centering_by_ps_matches_an_unpenalized_intercept_on_a_full_fit():
    """Determined system: both centering routes must land on the same coefficients.

    Fit the whole library, so the Lasso optimum is unique. On an underdetermined fit the
    two routes may pick different members of a tied optimal set.
    """
    L = 5
    ps, _, df = sparse_landscape(L, n_nonzero=3, seed=4)
    full = (
        np.stack([np.fromiter(g, dtype=np.int8) for g in df["genotype"]]),
        df["fitness"].to_numpy(float),
    )
    train = test = full
    kw = dict(lambda_selection="grid", lambdas=[1e-8])
    centered = ewht.run_cs_experiment(train, test, ps, center_by_ps=True, **kw)
    intercepted = ewht.run_cs_experiment(train, test, ps, center_by_ps=False, **kw)
    assert np.allclose(centered.coeffs[1:], intercepted.coeffs[1:], atol=1e-3)
    assert centered.test_metrics["r2"] == pytest.approx(
        intercepted.test_metrics["r2"], abs=1e-3
    )


def test_split_is_uniform_without_replacement_and_holds_out_the_rest():
    df = pd.DataFrame(
        {"genotype": [f"{i:04b}" for i in range(16)], "fitness": np.arange(16.0)}
    )
    in_bin = np.zeros(16, dtype=bool)
    in_bin[:10] = True
    train, test = ewht.split_bin_train_test(df, 4, in_bin=in_bin, random_state=0)
    assert len(train) == 4 and len(test) == 6
    assert train["genotype"].is_unique
    assert not set(train["genotype"]) & set(test["genotype"])
    assert set(train["genotype"]) | set(test["genotype"]) == set(df["genotype"][:10])


def test_training_sets_are_nested_when_a_pool_is_shared():
    df = pd.DataFrame(
        {"genotype": [f"{i:05b}" for i in range(32)], "fitness": np.arange(32.0)}
    )
    small, _ = ewht.split_bin_train_test(df, 5, pool_size=10, random_state=7)
    large, _ = ewht.split_bin_train_test(df, 10, pool_size=10, random_state=7)
    assert set(small["genotype"]).issubset(set(large["genotype"]))


def test_split_rejects_a_pool_that_swallows_the_bin():
    df = pd.DataFrame({"genotype": ["00", "01", "10", "11"], "fitness": [0.0, 1, 2, 3]})
    with pytest.raises(ValueError):
        ewht.split_bin_train_test(df, 4)


def test_bin_column_may_be_named():
    df = pd.DataFrame(
        {
            "genotype": [f"{i:04b}" for i in range(16)],
            "fitness": np.arange(16.0),
            "DMS_score_bin_plm": [1] * 8 + [0] * 8,
        }
    )
    train, test = ewht.split_bin_train_test(df, 3, in_bin="DMS_score_bin_plm", random_state=1)
    assert len(train) == 3 and len(test) == 5


def test_legacy_centering_and_grid_are_still_reachable():
    """The pre-0.1 centering and lambda grid remain available as explicit flags.

    The stable-norm design matrix is not: compressed sensing is expectation-norm only.
    The fit here is heavily underdetermined (100 rows, 2048 columns) and its optimum is
    not unique, so only the regime is checked, not an exact alpha or R2.
    """
    from ewht.examples import CR6261_MUTANTS, CR6261_POSITIONS, CR6261_WT_SEQUENCE

    df = ewht.genotypes_from_dataframe(
        ewht.load_example(),
        CR6261_POSITIONS,
        wt_sequence=CR6261_WT_SEQUENCE,
        mutants=CR6261_MUTANTS,
    )
    with ewht.example_msa() as msa:
        ps = ewht.get_ps(CR6261_WT_SEQUENCE, CR6261_POSITIONS, CR6261_MUTANTS, msa=msa)
        train, test = ewht.sample_evolutionary_sequences(
            df,
            ps,
            msa=msa,
            positions=CR6261_POSITIONS,
            wt_sequence=CR6261_WT_SEQUENCE,
            mutants=CR6261_MUTANTS,
            fraction=0.75,
            train_n=100,
            random_state=0,
        )
    result = ewht.run_cs_experiment(
        train,
        test,
        ps,
        center_by_ps=True,
        lambda_selection="grid",
        random_state=0,
    )
    assert result.best_lambda in ewht.cs.DEFAULT_LAMBDAS
    assert result.test_metrics["r2"] > 0.7
    with pytest.raises(TypeError):      # removed on purpose
        ewht.run_cs_experiment(train, test, ps, basis_norm="stable")


def test_ewht_beats_wht_on_the_bundled_landscape():
    """The paper's claim, on its own example: fewer measurements go further under ps."""
    from ewht.examples import CR6261_MUTANTS, CR6261_POSITIONS, CR6261_WT_SEQUENCE

    df = ewht.genotypes_from_dataframe(
        ewht.load_example(),
        CR6261_POSITIONS,
        wt_sequence=CR6261_WT_SEQUENCE,
        mutants=CR6261_MUTANTS,
    )
    ps, in_bin = ewht.get_ps_from_scores(df, "plm_score", return_bin=True)
    measured = (df["estimated_fitness"] == 0).to_numpy()
    train, test = ewht.split_bin_train_test(
        df[measured].reset_index(drop=True),
        100,
        in_bin=in_bin[measured],
        random_state=0,
    )
    scores = {
        basis: ewht.run_cs_experiment(train, test, ps, basis=basis).test_metrics["r2"]
        for basis in ("eWHT", "WHT")
    }
    assert scores["eWHT"] > scores["WHT"]
