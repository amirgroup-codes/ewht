from __future__ import annotations

from typing import Union

import numpy as np
import pandas as pd

from ewht._columns import resolve_column, resolve_genotype_column
from ewht._genotype import infer_genotype_length, validate_genotype_series

SeqData = Union[pd.DataFrame, tuple[np.ndarray, np.ndarray]]


def parse_seq_data(data: SeqData) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse binary genotypes and fitness values from a DataFrame or tuple.

    Inputs:
    - ``data``: pandas DataFrame with genotype + fitness columns, or
      ``(bin_seqs, y)`` with ``bin_seqs`` ``(N, L)`` int8 and ``y`` ``(N,)`` float.

    Outputs:
    - ``bin_seqs``: ``(N, L)`` int8 array.
    - ``y``: ``(N,)`` float array (duplicate genotypes averaged).
    """
    if isinstance(data, tuple):
        if len(data) != 2:
            raise ValueError("Tuple input must be (bin_seqs, y).")
        bin_seqs, y = data
        bin_seqs = np.asarray(bin_seqs, dtype=np.int8)
        y = np.asarray(y, dtype=float)
        if bin_seqs.ndim != 2:
            raise ValueError("bin_seqs must be a 2D array of shape (N, L).")
        if y.ndim != 1 or len(y) != len(bin_seqs):
            raise ValueError("y must be a 1D array with one value per sequence.")
        return bin_seqs, y

    if isinstance(data, pd.DataFrame):
        if data.empty:
            raise ValueError("Input DataFrame is empty.")

        genotype_col = resolve_genotype_column(data)
        fitness_col = resolve_column(data, ["y", "fitness", "score"], purpose="fitness")

        work = data[[genotype_col, fitness_col]].copy()
        L = infer_genotype_length(work[genotype_col])
        work[genotype_col] = validate_genotype_series(work[genotype_col], L=L)
        work[fitness_col] = pd.to_numeric(work[fitness_col], errors="coerce")
        if work[fitness_col].isna().any():
            raise ValueError(f"Non-numeric values found in '{fitness_col}'.")

        grouped = work.groupby(genotype_col, as_index=False)[fitness_col].mean()
        bin_seqs = np.stack(
            [np.fromiter(g, dtype=np.int8) for g in grouped[genotype_col].to_numpy()]
        )
        y = grouped[fitness_col].to_numpy(dtype=float)
        return bin_seqs, y

    raise TypeError("Expected a pandas DataFrame or (bin_seqs, y) tuple.")
