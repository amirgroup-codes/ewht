"""Evolutionary probabilities from a protein language model scored over a measured library.

This is the method used in "Protein Fitness Landscapes Are Simpler Under Evolutionary
Distributions":

1. **Score the whole landscape** with a protein language model. Two are built in, chosen
   with ``model=``:

   - ``"esm2"`` (default): ESM2 wildtype-marginal LLR. One forward pass over the
     wild-type sequence; a variant's score is the sum of its substitutions' LLRs. Needs
     ``wt_sequence``, ``positions`` and ``mutants``.
   - ``"antiberty"``: AntiBERTy pseudo-log-likelihood. Each variant is scored on its own
     sequence, so it needs the variant sequences (a sequence column, or ``wt_sequence``
     plus ``positions`` and ``mutants`` to rebuild them). Antibodies only.

2. **Split the library at the median score.** The upper half is the pLM bin, recorded as
   ``plm_score_bin`` (1 in the bin, 0 below), next to the score itself in ``plm_score``.

3. **Read off ``p_i``** as the fraction of bin variants carrying a mutation at site ``i``,
   clipped to ``ps_bounds``.

Because the probabilities come from a bin of variants, this needs the measured library,
not just the wild-type sequence. ``ewht.ps.get_ps`` remains for single-sequence estimates
(an MSA column, or ESM2 masked marginals); it does not reproduce the paper.

Scores from a model this package does not wrap can be passed straight in with
``scores=``, which also lets an expensive scoring run be cached and reused.
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ewht.ps._constants import (
    ESM_MODEL_NAME,
    PLM_DEFAULT_BOUNDS,
    X_MUTANT_AAS,
)
from ewht.ps._parse import normalize_mutants, validate_sequence

MODELS = ("esm2", "antiberty")
PLM_SCORE_COLUMN = "plm_score"
PLM_BIN_COLUMN = "plm_score_bin"


def _genotype_helpers():
    """Imported lazily: ewht._genotype reads constants from this subpackage."""
    from ewht._columns import resolve_column, resolve_genotype_column
    from ewht._genotype import validate_genotype_series

    return resolve_column, resolve_genotype_column, validate_genotype_series


# ---------------------------------------------------------------- input handling
def _read_landscape(data: pd.DataFrame | str | Path) -> pd.DataFrame:
    """Accept a DataFrame or a path to a CSV of the landscape."""
    if isinstance(data, pd.DataFrame):
        if data.empty:
            raise ValueError("Input DataFrame is empty.")
        return data.copy()
    if isinstance(data, (str, Path)):
        path = Path(data)
        if not path.exists():
            raise FileNotFoundError(f"Landscape CSV not found: {path}")
        frame = pd.read_csv(path, dtype={"genotype": str})
        if frame.empty:
            raise ValueError(f"Landscape CSV is empty: {path}")
        return frame
    raise TypeError("data must be a pandas DataFrame or a path to a CSV file.")


def _ensure_genotypes(
    df: pd.DataFrame,
    *,
    genotype_col: str | None,
    wt_sequence: str | None,
    positions: Sequence[int | str] | None,
    mutants: Sequence[str | None] | Mapping[int | str, str] | None,
) -> tuple[pd.DataFrame, str]:
    """Return the frame with a validated genotype column, encoding one if absent."""
    _, resolve_genotype_column, validate_genotype_series = _genotype_helpers()
    if genotype_col is None:
        try:
            genotype_col = resolve_genotype_column(df)
        except KeyError:
            genotype_col = None
    if genotype_col is not None and genotype_col in df.columns:
        L = len(positions) if positions is not None else None
        df = df.copy()
        df[genotype_col] = validate_genotype_series(df[genotype_col], L=L)
        return df, genotype_col

    if wt_sequence is None or positions is None or mutants is None:
        raise ValueError(
            "No genotype column found. Provide one, or give wt_sequence, positions and "
            "mutants so genotypes can be encoded from the variant sequences."
        )
    from ewht.data import genotypes_from_dataframe

    return genotypes_from_dataframe(df, list(positions), wt_sequence, mutants), "genotype"


def variant_sequences(
    df: pd.DataFrame,
    wt_sequence: str,
    positions: Sequence[int | str],
    mutants: Sequence[str | None] | Mapping[int | str, str],
    *,
    genotype_col: str = "genotype",
) -> list[list[str]]:
    """
    Rebuild each variant's amino-acid sequence from its genotype.

    Inputs:
    - ``df``: pandas DataFrame with a 0/1 genotype column.
    - ``wt_sequence``: wild-type amino-acid string.
    - ``positions``: length-``L`` list of 1-indexed positions; ``"NC"`` sites leave the
      sequence unchanged, since they are not residues.
    - ``mutants``: mutant amino acid per site. ``'X'`` marks an ambiguous residue standing
      for any of V/L/I.
    - ``genotype_col``: genotype column name.

    Outputs:
    - a list of readings, each a length-``N`` list of sequences. One reading unless a
      mutant is ``'X'``, in which case there is one reading per alternative (every
      ambiguous site taking that amino acid), to be combined on the perplexity scale.
    """
    wt = validate_sequence(wt_sequence)
    mutant_list = normalize_mutants(list(positions), mutants)
    ambiguous = any(m is not None and m.upper() == "X" for m in mutant_list)
    readings = list(X_MUTANT_AAS) if ambiguous else [None]

    out: list[list[str]] = []
    for reading in readings:
        sequences: list[str] = []
        for genotype in df[genotype_col].astype(str):
            seq = list(wt)
            for bit, pos, mut in zip(genotype, positions, mutant_list):
                if bit != "1" or pos == "NC":
                    continue
                if mut is None:
                    raise ValueError(
                        f"mutants must name the mutant residue at position {pos} to "
                        "rebuild variant sequences; pass a sequence column instead."
                    )
                aa = reading if mut.upper() == "X" else mut.upper()
                seq[int(pos) - 1] = aa
            sequences.append("".join(seq))
        out.append(sequences)
    return out


# ---------------------------------------------------------------- scoring
def _score_esm2(df, genotype_col, wt_sequence, positions, mutants, **kw) -> np.ndarray:
    from ewht.ps.esm_marginal import score_wt_marginal

    if wt_sequence is None or positions is None:
        raise ValueError("model='esm2' needs wt_sequence and positions.")
    return score_wt_marginal(
        df[genotype_col].astype(str).tolist(),
        wt_sequence,
        positions,
        mutants,
        llr=kw.get("llr"),
        cache_folder=kw.get("cache_folder"),
        model_name=kw.get("model_name") or ESM_MODEL_NAME,
        device=kw.get("device"),
    )


def _score_antiberty(df, genotype_col, wt_sequence, positions, mutants, **kw) -> np.ndarray:
    from ewht.ps.antiberty import combine_readings, pseudo_log_likelihood

    sequence_col = kw.get("sequence_col")
    if sequence_col is None and wt_sequence is None:
        resolve_column, _, _ = _genotype_helpers()
        try:
            sequence_col = resolve_column(
                df, ["mutated_sequence", "variant_sequence", "sequence"], purpose="sequence"
            )
        except KeyError as exc:
            raise ValueError(
                "model='antiberty' scores each variant's own sequence: provide "
                "sequence_col, or wt_sequence with positions and mutants."
            ) from exc

    if sequence_col is not None:
        readings = [df[sequence_col].astype(str).str.strip().tolist()]
    else:
        if positions is None or mutants is None:
            raise ValueError(
                "model='antiberty' needs positions and mutants to rebuild sequences."
            )
        readings = variant_sequences(
            df, wt_sequence, positions, mutants, genotype_col=genotype_col
        )

    plls = np.stack(
        [
            pseudo_log_likelihood(
                seqs,
                device=kw.get("device"),
                batch_sequences=kw.get("batch_sequences"),
                verbose=kw.get("verbose", False),
            )
            for seqs in readings
        ]
    )
    return combine_readings(plls if len(plls) > 1 else plls[0])


_SCORERS = {"esm2": _score_esm2, "antiberty": _score_antiberty}


def score_landscape(
    data: pd.DataFrame | str | Path,
    model: str = "esm2",
    *,
    wt_sequence: str | None = None,
    positions: Sequence[int | str] | None = None,
    mutants: Sequence[str | None] | Mapping[int | str, str] | None = None,
    genotype_col: str | None = None,
    sequence_col: str | None = None,
    scores: np.ndarray | None = None,
    higher_is_fitter: bool = True,
    threshold: float | None = None,
    llr: pd.DataFrame | None = None,
    cache_folder: str | Path | None = None,
    model_name: str | None = None,
    device: str | None = None,
    batch_sequences: int | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Score a fitness landscape with a protein language model and split it at the median.

    Inputs:
    - ``data``: the landscape, as a pandas DataFrame or a path to a CSV. Needs a 0/1
      genotype column, or ``wt_sequence`` with ``positions`` and ``mutants`` so one can
      be encoded from the variant sequences.
    - ``model``: ``'esm2'`` (wildtype-marginal LLR, the default) or ``'antiberty'``
      (pseudo-log-likelihood, antibodies only). Ignored when ``scores`` is given.
    - ``wt_sequence``, ``positions``, ``mutants``: the library definition. ``positions``
      are 1-indexed; ``"NC"`` marks a non-coding site.
    - ``genotype_col``, ``sequence_col``: optional column names.
    - ``scores``: ``(N,)`` precomputed scores aligned with the rows, which skips the
      model entirely. Use this to reuse an expensive scoring run.
    - ``higher_is_fitter``: set ``False`` when model score runs against fitness; the sign
      is flipped before the median split, so ``plm_score`` is always oriented with
      higher = treated as fitter. ``spearman_score_vs_fitness`` tells you which way round
      your data is.
    - ``threshold``: optional cut to use instead of the median.
    - ``llr``: precomputed wildtype LLR matrix for ``'esm2'``.
    - ``cache_folder``, ``model_name``, ``device``, ``batch_sequences``, ``verbose``:
      passed to the scorer.

    Outputs:
    - a copy of the landscape with a validated ``genotype`` column plus
      ``plm_score`` (float, oriented) and ``plm_score_bin`` (1 at or above the median
      score, else 0).
    """
    df = _read_landscape(data)
    df, genotype_col = _ensure_genotypes(
        df,
        genotype_col=genotype_col,
        wt_sequence=wt_sequence,
        positions=positions,
        mutants=mutants,
    )

    if scores is None:
        key = str(model).lower()
        if key not in _SCORERS:
            raise ValueError(f"model must be one of {MODELS}, got {model!r}.")
        values = _SCORERS[key](
            df,
            genotype_col,
            wt_sequence,
            positions,
            mutants,
            llr=llr,
            cache_folder=cache_folder,
            model_name=model_name,
            device=device,
            sequence_col=sequence_col,
            batch_sequences=batch_sequences,
            verbose=verbose,
        )
    else:
        values = np.asarray(scores, dtype=float)
        if values.shape != (len(df),):
            raise ValueError(
                f"scores must have one value per row: got {values.shape} for {len(df)} rows."
            )

    values = np.asarray(values, dtype=float)
    if not higher_is_fitter:
        values = -values
    df[PLM_SCORE_COLUMN] = values
    df[PLM_BIN_COLUMN] = plm_bin(values, threshold=threshold).astype(int)
    return df


# ---------------------------------------------------------------- bin and ps
def plm_bin(scores: np.ndarray, *, threshold: float | None = None) -> np.ndarray:
    """
    Split a scored library at its median into the pLM bin.

    Inputs:
    - ``scores``: ``(N,)`` float variant scores, higher = more fit.
    - ``threshold``: optional cut to use instead of the median.

    Outputs:
    - ``(N,)`` bool array, ``True`` for variants in the bin (``score >= threshold``).
      Non-finite scores are excluded.
    """
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1:
        raise ValueError("scores must be a 1D array.")
    finite = np.isfinite(scores)
    if not finite.any():
        raise ValueError("scores contains no finite values.")
    cut = float(np.median(scores[finite])) if threshold is None else float(threshold)
    return finite & (scores >= cut)


def ps_from_bin(
    bin_seqs: np.ndarray,
    in_bin: np.ndarray,
    *,
    ps_bounds: tuple[float, float] | Sequence[float] | None = None,
) -> np.ndarray:
    """
    Per-site mutation frequency among the variants in a bin.

    Inputs:
    - ``bin_seqs``: ``(N, L)`` array of 0/1 genotypes (``0`` = wild-type, ``1`` = mutant).
    - ``in_bin``: ``(N,)`` bool mask selecting the bin.
    - ``ps_bounds``: ``(low, high)`` clip interval; defaults to ``(0.05, 0.95)``.

    Outputs:
    - ``(L,)`` float64 array of clipped mutation probabilities.
    """
    bin_seqs = np.asarray(bin_seqs, dtype=np.int8)
    in_bin = np.asarray(in_bin, dtype=bool)
    if bin_seqs.ndim != 2:
        raise ValueError("bin_seqs must be a 2D array of shape (N, L).")
    if in_bin.shape != (len(bin_seqs),):
        raise ValueError("in_bin must have one entry per sequence.")
    if not in_bin.any():
        raise ValueError("The bin is empty: no variants to estimate p_i from.")
    low, high = tuple(PLM_DEFAULT_BOUNDS if ps_bounds is None else ps_bounds)
    if low >= high:
        raise ValueError(f"ps_bounds must satisfy low < high, got {(low, high)}.")
    return np.clip(bin_seqs[in_bin].mean(axis=0), low, high).astype(np.float64)


def _genotype_matrix(df: pd.DataFrame, genotype_col: str | None) -> np.ndarray:
    _, resolve_genotype_column, validate_genotype_series = _genotype_helpers()
    if genotype_col is None:
        genotype_col = resolve_genotype_column(df)
    genotypes = validate_genotype_series(df[genotype_col])
    return np.stack([np.fromiter(g, dtype=np.int8) for g in genotypes.to_numpy()])


def ps_from_scored_landscape(
    df: pd.DataFrame,
    *,
    genotype_col: str | None = None,
    bin_col: str = PLM_BIN_COLUMN,
    ps_bounds: tuple[float, float] | Sequence[float] | None = None,
) -> np.ndarray:
    """
    Evolutionary probabilities from a landscape already scored and binned.

    Inputs:
    - ``df``: the frame returned by ``score_landscape``.
    - ``genotype_col``: optional genotype column name.
    - ``bin_col``: the 0/1 bin column; defaults to ``plm_score_bin``.
    - ``ps_bounds``: ``(low, high)`` clip interval; defaults to ``(0.05, 0.95)``.

    Outputs:
    - ``(L,)`` float64 array of mutation probabilities.
    """
    if bin_col not in df.columns:
        raise KeyError(
            f"No '{bin_col}' column: run score_landscape first, or pass bin_col."
        )
    in_bin = pd.to_numeric(df[bin_col], errors="coerce").fillna(0).astype(int).to_numpy() == 1
    return ps_from_bin(_genotype_matrix(df, genotype_col), in_bin, ps_bounds=ps_bounds)


def get_ps_from_scores(
    df: pd.DataFrame,
    scores: np.ndarray | str,
    *,
    genotype_col: str | None = None,
    higher_is_fitter: bool = True,
    threshold: float | None = None,
    ps_bounds: tuple[float, float] | Sequence[float] | None = None,
    return_bin: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """
    Evolutionary probabilities from a library scored by any model.

    Splits the library at the median score and sets ``p_i`` to the mutation frequency at
    site ``i`` among the upper half.

    Inputs:
    - ``df``: pandas DataFrame with a genotype column, one row per measured variant.
    - ``scores``: ``(N,)`` float array aligned with ``df`` rows, or the name of a column
      of ``df`` holding them.
    - ``genotype_col``: optional genotype column name.
    - ``higher_is_fitter``: set ``False`` when the score runs against fitness.
    - ``threshold``: optional cut to use instead of the median.
    - ``ps_bounds``: ``(low, high)`` clip interval; defaults to ``(0.05, 0.95)``.
    - ``return_bin``: if ``True``, also return the bin mask.

    Outputs:
    - ``(L,)`` float64 array of mutation probabilities, and the ``(N,)`` bool bin mask
      when ``return_bin=True``.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    if isinstance(scores, str):
        values = pd.to_numeric(df[scores], errors="coerce").to_numpy(dtype=float)
    else:
        values = np.asarray(scores, dtype=float)
        if values.shape != (len(df),):
            raise ValueError(
                f"scores must have one value per row: got {values.shape} for {len(df)} rows."
            )
    if not higher_is_fitter:
        values = -values

    in_bin = plm_bin(values, threshold=threshold)
    ps = ps_from_bin(_genotype_matrix(df, genotype_col), in_bin, ps_bounds=ps_bounds)
    return (ps, in_bin) if return_bin else ps


def get_ps_from_library(
    data: pd.DataFrame | str | Path,
    model: str = "esm2",
    *,
    wt_sequence: str | None = None,
    positions: Sequence[int | str] | None = None,
    mutants: Sequence[str | None] | Mapping[int | str, str] | None = None,
    genotype_col: str | None = None,
    sequence_col: str | None = None,
    scores: np.ndarray | None = None,
    higher_is_fitter: bool = True,
    threshold: float | None = None,
    ps_bounds: tuple[float, float] | Sequence[float] | None = None,
    llr: pd.DataFrame | None = None,
    cache_folder: str | Path | None = None,
    model_name: str | None = None,
    device: str | None = None,
    batch_sequences: int | None = None,
    verbose: bool = False,
    return_frame: bool = False,
) -> np.ndarray | tuple[np.ndarray, pd.DataFrame]:
    """
    Score a landscape with a protein language model and estimate ``ps`` from the bin.

    One call for the paper's whole ``ps`` procedure: score, median split, per-site
    mutation frequency in the upper half.

    Inputs:
    - as ``score_landscape``, plus ``ps_bounds`` (clip interval, default
      ``(0.05, 0.95)``) and ``return_frame``.

    Outputs:
    - ``(L,)`` float64 array of mutation probabilities; with ``return_frame=True``, also
      the scored landscape carrying ``plm_score`` and ``plm_score_bin``.
    """
    scored = score_landscape(
        data,
        model,
        wt_sequence=wt_sequence,
        positions=positions,
        mutants=mutants,
        genotype_col=genotype_col,
        sequence_col=sequence_col,
        scores=scores,
        higher_is_fitter=higher_is_fitter,
        threshold=threshold,
        llr=llr,
        cache_folder=cache_folder,
        model_name=model_name,
        device=device,
        batch_sequences=batch_sequences,
        verbose=verbose,
    )
    ps = ps_from_scored_landscape(scored, genotype_col=genotype_col, ps_bounds=ps_bounds)
    return (ps, scored) if return_frame else ps


def spearman_score_vs_fitness(
    df: pd.DataFrame,
    scores: np.ndarray | str = PLM_SCORE_COLUMN,
    *,
    fitness_col: str | None = None,
) -> float:
    """
    Spearman correlation of model score against measured fitness.

    A negative value is the sign to pass ``higher_is_fitter=False`` when estimating
    ``ps``.

    Inputs:
    - ``df``: pandas DataFrame with a fitness column.
    - ``scores``: ``(N,)`` float array aligned with ``df`` rows, or a column name;
      defaults to ``plm_score``.
    - ``fitness_col``: optional fitness column name.

    Outputs:
    - float Spearman rho over the rows where both values are finite (``nan`` if fewer
      than two such rows).
    """
    from scipy.stats import spearmanr

    resolve_column, _, _ = _genotype_helpers()
    if fitness_col is None:
        fitness_col = resolve_column(df, ["y", "fitness", "score"], purpose="fitness")
    fitness = pd.to_numeric(df[fitness_col], errors="coerce").to_numpy(dtype=float)
    values = (
        pd.to_numeric(df[scores], errors="coerce").to_numpy(dtype=float)
        if isinstance(scores, str)
        else np.asarray(scores, dtype=float)
    )
    mask = np.isfinite(fitness) & np.isfinite(values)
    if mask.sum() < 2:
        return float("nan")
    rho = spearmanr(fitness[mask], values[mask]).statistic
    return float("nan") if np.isnan(rho) else float(rho)
