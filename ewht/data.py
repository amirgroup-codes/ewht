from __future__ import annotations

import warnings
from pathlib import Path
from typing import Mapping

import numpy as np
import pandas as pd

from ewht._columns import resolve_column
from ewht._genotype import (
    build_landscape_weights,
    encode_genotype_from_sequence,
    index_to_genotype,
    mutant_matches,
    normalize_genotype_string,
    validate_genotype_series,
)
from ewht.cs import center_targets
from ewht.ps._parse import normalize_mutants
from ewht.ps.msa import read_alignment


def genotypes_from_dataframe(
    df: pd.DataFrame,
    positions: list[int | str],
    wt_sequence: str,
    mutants: list[str | None] | Mapping[int | str, str],
    *,
    sequence_col: str | None = None,
    fitness_col: str | None = None,
    mutant_col: str | None = None,
) -> pd.DataFrame:
    """
    Encode binary genotypes from amino-acid sequences and validate the result.

    Builds a ``genotype`` column (``0`` = wild-type, ``1`` = mutant at each site).
    Duplicate genotypes are averaged with a warning. Ambiguous residues raise
    ``ValueError``.

    Inputs:
    - ``df``: pandas DataFrame with sequence and fitness columns.
    - ``positions``: length-``L`` list of 1-indexed site positions or ``"NC"`` tokens.
    - ``wt_sequence``: wild-type amino-acid string.
    - ``mutants``: length-``L`` list or dict mapping sites to mutant amino acids.
      Required; pass explicit genotypes manually if your mapping is custom.
    - ``sequence_col``, ``fitness_col``, ``mutant_col``: optional column names.

    Outputs:
    - pandas DataFrame with a validated ``genotype`` column (``str`` of ``0``/``1`` bits)
      and numeric fitness, one row per unique genotype.
    """
    if mutants is None:
        raise ValueError(
            "mutants must be provided. If your genotype mapping is custom, "
            "add a genotype column directly instead of using this helper."
        )

    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    work = df.copy()
    mutant_list = normalize_mutants(positions, mutants)
    wt_sequence = wt_sequence.strip().upper()
    L = len(positions)

    if sequence_col is None:
        sequence_col = resolve_column(
            work,
            ["mutated_sequence", "genotype_sequence", "sequence"],
            purpose="sequence",
        )
    if fitness_col is None:
        fitness_col = resolve_column(work, ["y", "fitness", "score"], purpose="fitness")
    if mutant_col is None:
        try:
            mutant_col = resolve_column(work, ["mutant"], purpose="mutant")
        except KeyError:
            mutant_col = None

    encoded: list[str] = []
    for _, row in work.iterrows():
        seq = str(row[sequence_col]).strip().upper()
        mutant_label = str(row[mutant_col]) if mutant_col is not None else None
        try:
            genotype = encode_genotype_from_sequence(
                seq,
                positions,
                mutant_list,
                wt_sequence,
                mutant_label=mutant_label,
            )
        except ValueError as exc:
            warnings.warn(str(exc), stacklevel=2)
            raise
        encoded.append(genotype)
    work["genotype"] = encoded

    work[fitness_col] = pd.to_numeric(work[fitness_col], errors="coerce")
    if work[fitness_col].isna().any():
        raise ValueError(f"Non-numeric values found in fitness column '{fitness_col}'.")

    if work["genotype"].duplicated(keep=False).any():
        n_dup = int(work.loc[work["genotype"].duplicated(keep=False), "genotype"].nunique())
        warnings.warn(
            f"Averaging fitness for {n_dup} duplicate genotype(s).",
            stacklevel=2,
        )
        agg = {fitness_col: "mean"}
        for col in work.columns:
            if col not in {fitness_col, "genotype"}:
                agg[col] = "first"
        work = work.groupby("genotype", as_index=False).agg(agg)

    work["genotype"] = validate_genotype_series(work["genotype"], L=L)
    return work.reset_index(drop=True)


def _msa_neighborhood_mask(
    ps: np.ndarray,
    msa: str | Path,
    positions: list[int | str],
    wt_sequence: str,
    mutants: list[str | None],
) -> np.ndarray:
    sequences = read_alignment(msa)
    L = len(ps)
    int_positions = [(idx, pos) for idx, pos in enumerate(positions) if pos != "NC"]
    skeletons: set[tuple[int, ...]] = set()

    for seq in sequences:
        if any(not isinstance(pos, int) or len(seq) < pos for _, pos in int_positions):
            continue
        genotype = [0] * L
        for idx, pos in int_positions:
            aa = seq[pos - 1].upper()
            wt = wt_sequence[pos - 1].upper()
            mut = mutants[idx]
            if mut is None:
                continue
            genotype[idx] = 1 if (mutant_matches(aa, mut) and aa != wt) else 0
        skeletons.add(tuple(genotype))

    mask = np.zeros(1 << L, dtype=bool)
    for global_idx in range(1 << L):
        genotype_tuple = tuple(int(b) for b in format(global_idx, f"0{L}b"))
        if genotype_tuple in skeletons:
            mask[global_idx] = True
            continue
        for skel in skeletons:
            if sum(g1 != g2 for g1, g2 in zip(genotype_tuple, skel)) <= 1:
                mask[global_idx] = True
                break
    return mask


def sample_evolutionary_sequences(
    df: pd.DataFrame,
    ps: np.ndarray,
    *,
    genotype_col: str | None = None,
    msa: str | Path | None = None,
    positions: list[int | str] | None = None,
    wt_sequence: str | None = None,
    mutants: list[str | None] | Mapping[int | str, str] | None = None,
    fraction: float = 0.75,
    train_n: int | None = None,
    random_state: int = 42,
) -> pd.DataFrame | tuple[pd.DataFrame, pd.DataFrame]:
    """
    Subsample a combinatorial landscape using evolutionary weights.

    Draws genotypes from the full ``2^L`` landscape with probability proportional to
    the product Bernoulli measure defined by ``ps``. Optionally restricts draws to
    an MSA neighborhood (skeleton genotypes and Hamming distance ≤ 1).

    Inputs:
    - ``df``: pandas DataFrame with a ``genotype`` column and fitness values.
    - ``ps``: ``(L,)`` float mutation probabilities.
    - ``msa``: optional path to ``.fasta``/``.a2m`` alignment; requires
      ``positions``, ``wt_sequence``, and ``mutants``.
    - ``fraction``: fraction of ``len(df)`` draws with replacement from the landscape
      (default ``0.75``); unique drawn genotypes present in ``df`` are kept.
    - ``train_n``: if set, split the subsample into ``train_n`` training rows and
      the remainder as test.
    - ``random_state``: int RNG seed.

    Outputs:
    - subsampled DataFrame, or ``(train_df, test_df)`` when ``train_n`` is provided.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")

    ps = np.asarray(ps, dtype=float)
    work = df.copy()

    if genotype_col is None:
        genotype_col = resolve_column(work, ["genotype"], purpose="genotype")
    work[genotype_col] = validate_genotype_series(work[genotype_col], L=len(ps))

    L = len(ps)
    weights = build_landscape_weights(ps)
    if msa is not None:
        if positions is None or wt_sequence is None or mutants is None:
            raise ValueError(
                "positions, wt_sequence, and mutants are required when msa is provided."
            )
        mutant_list = normalize_mutants(positions, mutants)
        mask = _msa_neighborhood_mask(ps, msa, positions, wt_sequence.strip().upper(), mutant_list)
        weights = weights * mask
        if np.sum(weights) <= 0:
            warnings.warn(
                "MSA neighborhood mask removed all weight; falling back to unmasked ps weights.",
                stacklevel=2,
            )
            weights = build_landscape_weights(ps)
    weights = weights / np.sum(weights)

    rng = np.random.default_rng(random_state)
    num_draws = int(fraction * len(work))
    sampled_indices = rng.choice(len(weights), size=num_draws, replace=True, p=weights)
    unique_indices = np.unique(sampled_indices)

    selected_genotypes = {index_to_genotype(int(idx), L) for idx in unique_indices}
    subsampled = work[work[genotype_col].isin(selected_genotypes)].copy().reset_index(drop=True)

    if train_n is None:
        return subsampled

    if train_n < 1:
        raise ValueError("train_n must be a positive integer.")
    if len(subsampled) < 2:
        raise ValueError("Need at least two rows in the subsampled DataFrame to split train/test.")
    if train_n >= len(subsampled):
        raise ValueError(
            f"train_n ({train_n}) must be smaller than the subsampled DataFrame size ({len(subsampled)})."
        )

    perm = rng.permutation(len(subsampled))
    train_df = subsampled.iloc[perm[:train_n]].reset_index(drop=True)
    test_df = subsampled.iloc[perm[train_n:]].reset_index(drop=True)
    return train_df, test_df


def split_bin_train_test(
    df: pd.DataFrame,
    n_train: int,
    *,
    in_bin: np.ndarray | str | None = None,
    pool_size: int | None = None,
    random_state: int = 0,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Train/test split inside a bin, as in the paper's compressed-sensing experiments.

    Draws a pool of measured variants uniformly without replacement from the bin, takes
    the first ``n_train`` of it as the training set, and holds out the whole bin minus
    that pool as the test set. Drawing one pool per seed and slicing its prefix means the
    training sets for different ``n_train`` are nested and every basis sees identical
    data, which is what makes eWHT and WHT comparable at fixed ``n``.

    Inputs:
    - ``df``: pandas DataFrame of measured variants, one row each.
    - ``n_train``: number of training rows.
    - ``in_bin``: ``(N,)`` bool mask, or the name of a 0/1 column, selecting the bin;
      ``None`` uses every row. Pass the mask from ``ewht.ps.get_ps_from_library``.
    - ``pool_size``: size of the drawn pool; defaults to ``n_train``. Set it to the
      largest ``n_train`` of a sweep and slice the training set yourself to get nested
      training sets, or call this once per ``n_train`` with the same ``pool_size``.
    - ``random_state``: int seed for the draw.

    Outputs:
    - ``(train_df, test_df)``, both with the original columns and a reset index.
    """
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    if n_train < 1:
        raise ValueError("n_train must be a positive integer.")

    if in_bin is None:
        mask = np.ones(len(df), dtype=bool)
    elif isinstance(in_bin, str):
        mask = pd.to_numeric(df[in_bin], errors="coerce").fillna(0).astype(int).to_numpy() == 1
    else:
        mask = np.asarray(in_bin, dtype=bool)
        if mask.shape != (len(df),):
            raise ValueError("in_bin must have one entry per row of df.")

    pool_rows = np.flatnonzero(mask)
    size = n_train if pool_size is None else int(pool_size)
    if size < n_train:
        raise ValueError(f"pool_size ({size}) must be at least n_train ({n_train}).")
    if size >= len(pool_rows):
        raise ValueError(
            f"pool_size ({size}) must be smaller than the bin ({len(pool_rows)} variants), "
            "so that held-out variants remain."
        )

    rng = np.random.default_rng(random_state)
    pool = rng.choice(pool_rows, size=size, replace=False)
    held_out = np.setdiff1d(pool_rows, pool, assume_unique=False)
    train_df = df.iloc[pool[:n_train]].reset_index(drop=True)
    test_df = df.iloc[held_out].reset_index(drop=True)
    return train_df, test_df


def subtract_mean(
    df: pd.DataFrame,
    ps: np.ndarray,
    *,
    genotype_col: str | None = None,
    fitness_col: str | None = None,
    basis: str = "eWHT",
    inplace: bool = False,
) -> pd.DataFrame:
    """
    Subtract a ps-weighted (eWHT) or uniform (WHT) mean from fitness values.

    Inputs:
    - ``df``: pandas DataFrame with ``genotype`` and fitness columns.
    - ``ps``: ``(L,)`` float mutation probabilities.
    - ``genotype_col``, ``fitness_col``: optional column names.
    - ``basis``: ``{'eWHT', 'WHT'}``.
    - ``inplace``: if ``True``, modify ``df`` in place.

    Outputs:
    - pandas DataFrame with centered fitness column (float).
    """
    out = df if inplace else df.copy()
    if genotype_col is None:
        genotype_col = resolve_column(out, ["genotype"], purpose="genotype")
    if fitness_col is None:
        fitness_col = resolve_column(out, ["y", "fitness", "score"], purpose="fitness")

    genotypes = validate_genotype_series(out[genotype_col], L=len(ps))
    y = pd.to_numeric(out[fitness_col], errors="coerce").to_numpy(dtype=float)
    if np.isnan(y).any():
        raise ValueError(f"Non-numeric values found in fitness column '{fitness_col}'.")

    bin_seqs = np.stack([np.fromiter(g, dtype=np.int8) for g in genotypes.to_numpy()])
    centered, _ = center_targets(y, bin_seqs, ps, basis=basis)
    out[fitness_col] = centered
    return out
