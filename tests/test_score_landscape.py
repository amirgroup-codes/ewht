"""The scoring entry point: model toggle, CSV input, and the plm_score / plm_score_bin frame."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import ewht
from ewht.examples import CR6261_MUTANTS, CR6261_POSITIONS, CR6261_WT_SEQUENCE
from ewht.ps.antiberty import combine_readings
from ewht.ps.plm import PLM_BIN_COLUMN, PLM_SCORE_COLUMN

CR6261_PS = np.array(
    [0.677, 0.627, 0.517, 0.597, 0.688, 0.545, 0.644, 0.651, 0.604, 0.563, 0.590]
)


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return ewht.load_example()


def test_scores_may_be_supplied_and_produce_both_columns(raw):
    scored = ewht.score_landscape(
        raw,
        wt_sequence=CR6261_WT_SEQUENCE,
        positions=CR6261_POSITIONS,
        mutants=CR6261_MUTANTS,
        scores=raw["plm_score"].to_numpy(float),
    )
    assert PLM_SCORE_COLUMN in scored and PLM_BIN_COLUMN in scored
    assert set(scored[PLM_BIN_COLUMN].unique()) <= {0, 1}
    assert "genotype" in scored.columns
    # every original column survives
    assert set(raw.columns) <= set(scored.columns)


def test_bin_is_the_upper_median_half(raw):
    scored = ewht.score_landscape(
        raw,
        wt_sequence=CR6261_WT_SEQUENCE,
        positions=CR6261_POSITIONS,
        mutants=CR6261_MUTANTS,
        scores=raw["plm_score"].to_numpy(float),
    )
    cut = scored[PLM_SCORE_COLUMN].median()
    in_bin = scored[PLM_BIN_COLUMN] == 1
    assert scored.loc[in_bin, PLM_SCORE_COLUMN].min() >= cut
    assert scored.loc[~in_bin, PLM_SCORE_COLUMN].max() <= cut
    assert in_bin.sum() == pytest.approx(len(scored) / 2, abs=1)


def test_ps_from_the_scored_frame_reproduces_the_paper(raw):
    ps, scored = ewht.get_ps_from_library(
        raw,
        wt_sequence=CR6261_WT_SEQUENCE,
        positions=CR6261_POSITIONS,
        mutants=CR6261_MUTANTS,
        scores=raw["plm_score"].to_numpy(float),
        return_frame=True,
    )
    assert np.allclose(ps, CR6261_PS, atol=5e-4)
    assert np.allclose(ewht.ps_from_scored_landscape(scored), ps)


def test_csv_path_is_accepted(tmp_path, raw):
    df = ewht.genotypes_from_dataframe(
        raw, CR6261_POSITIONS, wt_sequence=CR6261_WT_SEQUENCE, mutants=CR6261_MUTANTS
    )
    csv = tmp_path / "landscape.csv"
    df.to_csv(csv, index=False)
    ps = ewht.get_ps_from_library(csv, scores=df["plm_score"].to_numpy(float))
    assert np.allclose(ps, CR6261_PS, atol=5e-4)


def test_unknown_model_is_rejected(raw):
    with pytest.raises(ValueError, match="model must be one of"):
        ewht.score_landscape(raw, "esm3", wt_sequence=CR6261_WT_SEQUENCE,
                             positions=CR6261_POSITIONS, mutants=CR6261_MUTANTS)


def test_missing_library_definition_is_rejected(raw):
    with pytest.raises(ValueError, match="No genotype column"):
        ewht.score_landscape(raw, scores=np.zeros(len(raw)))


def test_model_names():
    assert ewht.MODELS == ("esm2", "antiberty")


# ---------------------------------------------------------------- AntiBERTy pieces
def test_variant_sequences_rebuilds_each_variant():
    wt = "ACDEF"
    df = pd.DataFrame({"genotype": ["000", "100", "011"]})
    seqs = ewht.variant_sequences(df, wt, [1, 3, 5], ["G", "W", "Y"])
    assert len(seqs) == 1
    assert seqs[0] == ["ACDEF", "GCDEF", "ACWEY"]


def test_variant_sequences_expands_an_ambiguous_mutant():
    wt = "ACDEF"
    df = pd.DataFrame({"genotype": ["10", "01"]})
    readings = ewht.variant_sequences(df, wt, [1, 5], ["X", "W"])
    assert len(readings) == 3                       # one per alternative for X
    assert [r[0][0] for r in readings] == ["V", "L", "I"]
    assert all(r[1] == "ACDEW" for r in readings)   # unaffected variant is identical


def test_nc_site_leaves_the_sequence_untouched():
    df = pd.DataFrame({"genotype": ["10", "01"]})
    seqs = ewht.variant_sequences(df, "ACDEF", ["NC", 5], [None, "W"])[0]
    assert seqs == ["ACDEF", "ACDEW"]


def test_combine_readings_averages_on_the_perplexity_scale():
    plls = np.array([[-1.0, -2.0], [-3.0, -2.0]])
    expected = -np.log(np.exp([1.0, 3.0]).mean()), -np.log(np.exp([2.0, 2.0]).mean())
    assert np.allclose(combine_readings(plls), expected)
    # a single reading passes through untouched
    assert np.allclose(combine_readings(np.array([-1.0, -2.0])), [-1.0, -2.0])


def test_antiberty_scoring_reproduces_a_cached_run():
    """Score a handful of variants with AntiBERTy and compare to the paper's cached PLLs."""
    pytest.importorskip("antiberty")
    import pandas as pd

    cache = __import__("pathlib").Path("/usr/scratch/pbias")
    pll_path = cache / "antiberty" / "pll" / "cr6261.npy"
    csv_path = cache / "data" / "processed_data" / "cr6261_h1_mean.csv"
    if not (pll_path.exists() and csv_path.exists()):
        pytest.skip("paper's cached AntiBERTy PLLs are not available here")

    from ewht.ps.antiberty import pseudo_log_likelihood

    cached = np.load(pll_path)
    frame = pd.read_csv(csv_path, dtype={"genotype": str}).head(8)
    try:
        scored = pseudo_log_likelihood(frame["mutated_sequence"].tolist())
    except OSError as exc:          # weights not cached and not downloadable here
        pytest.skip(f"AntiBERTy weights unavailable: {exc}")
    assert np.allclose(scored, cached[: len(frame)], atol=1e-5)
