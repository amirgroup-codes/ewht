"""The paper's ps method: score a library, split it at the median, read off p_i."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ewht
from ewht.ps._constants import PLM_DEFAULT_BOUNDS

# ps the paper reports for the bundled CR6261 H1-mean landscape.
CR6261_PS = np.array(
    [0.677, 0.627, 0.517, 0.597, 0.688, 0.545, 0.644, 0.651, 0.604, 0.563, 0.590]
)


@pytest.fixture(scope="module")
def example_df() -> pd.DataFrame:
    from ewht.examples import CR6261_MUTANTS, CR6261_POSITIONS, CR6261_WT_SEQUENCE

    return ewht.genotypes_from_dataframe(
        ewht.load_example(),
        CR6261_POSITIONS,
        wt_sequence=CR6261_WT_SEQUENCE,
        mutants=CR6261_MUTANTS,
    )


def test_reproduces_the_published_ps(example_df):
    ps = ewht.get_ps_from_scores(example_df, "plm_score")
    assert np.allclose(ps, CR6261_PS, atol=5e-4)


def test_bin_is_the_upper_median_half(example_df):
    ps, in_bin = ewht.get_ps_from_scores(example_df, "plm_score", return_bin=True)
    scores = example_df["plm_score"].to_numpy(float)
    assert in_bin.sum() == pytest.approx(len(example_df) / 2, abs=1)
    assert scores[in_bin].min() >= np.median(scores)
    assert len(ps) == 11


def test_ps_is_the_site_mutation_frequency_in_the_bin():
    bin_seqs = np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=np.int8)
    in_bin = np.array([True, True, True, False])
    ps = ewht.ps_from_bin(bin_seqs, in_bin, ps_bounds=(0.0001, 0.9999))
    assert np.allclose(ps, [2 / 3, 1 / 3])


def test_ps_bounds_are_applied():
    bin_seqs = np.array([[0, 1], [0, 1]], dtype=np.int8)
    ps = ewht.ps_from_bin(bin_seqs, np.array([True, True]))
    assert np.allclose(ps, PLM_DEFAULT_BOUNDS)
    tight = ewht.ps_from_bin(bin_seqs, np.array([True, True]), ps_bounds=(0.2, 0.8))
    assert np.allclose(tight, [0.2, 0.8])


def test_explicit_threshold(example_df):
    scores = example_df["plm_score"].to_numpy(float)
    _, in_bin = ewht.get_ps_from_scores(
        example_df, "plm_score", threshold=float(scores.max()), return_bin=True
    )
    assert in_bin.sum() >= 1 and in_bin.sum() < len(scores)


def test_scores_may_be_an_array(example_df):
    from_column = ewht.get_ps_from_scores(example_df, "plm_score")
    from_array = ewht.get_ps_from_scores(
        example_df, example_df["plm_score"].to_numpy(float)
    )
    assert np.allclose(from_column, from_array)


def test_wt_marginal_score_is_additive_and_zero_at_wildtype():
    """Scores come from one LLR matrix, so a double mutant is the sum of its singles."""
    llr = pd.DataFrame(
        {"A 1": [0.0, -1.0, -3.0], "C 2": [-2.0, 0.0, -4.0]},
        index=["A", "C", "D"],
    )
    scores = ewht.score_wt_marginal(
        ["00", "10", "01", "11"], "AC", [1, 2], ["D", "D"], llr=llr
    )
    assert np.allclose(scores, [0.0, -3.0, -4.0, -7.0])


def test_nc_sites_do_not_contribute():
    llr = pd.DataFrame({"A 1": [0.0, -2.0]}, index=["A", "D"])
    scores = ewht.score_wt_marginal(["00", "01"], "A", ["NC", 1], ["D", "D"], llr=llr)
    assert np.allclose(scores, [0.0, -2.0])


def test_ambiguous_mutant_combines_alternatives():
    llr = pd.DataFrame(
        {"A 1": [0.0, -1.0, -1.0, -1.0]}, index=["A", "V", "L", "I"]
    )
    value = ewht.site_llr(llr, 1, "X")
    assert np.isclose(value, np.log(3 * np.exp(-1.0)))


def test_rejects_misaligned_scores(example_df):
    with pytest.raises(ValueError):
        ewht.get_ps_from_scores(example_df, np.zeros(3))
    with pytest.raises(ValueError):
        ewht.ps_from_bin(np.zeros((4, 2), dtype=np.int8), np.zeros(4, dtype=bool))
