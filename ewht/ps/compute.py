from __future__ import annotations

from pathlib import Path
from typing import Mapping

import numpy as np

from ewht.ps._constants import ESM_DEFAULT_BOUNDS, MSA_DEFAULT_BOUNDS
from ewht.ps._parse import normalize_mutants, validate_sequence
from ewht.ps.esm import ps_from_esm
from ewht.ps.msa import ps_from_msa


def _clip_ps(ps: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
    low, high = bounds
    if low >= high:
        raise ValueError(f"ps_bounds must satisfy low < high, got {bounds}.")
    return np.clip(ps, low, high)


def get_ps(
    sequence: str,
    positions: list[int | str],
    mutants: list[str | None] | Mapping[int | str, str] | None = None,
    msa: str | Path | None = None,
    *,
    cache_folder: str | Path | None = None,
    ps_bounds: tuple[float, float] | list[float] | None = None,
) -> np.ndarray:
    """
    Estimate per-site mutation probabilities ``p(mutant)`` for selected positions.

    Uses an MSA (``.fasta``/``.a2m``) when ``msa`` is provided; otherwise queries
    ESM2-650M (requires optional ``ewht[esm]`` dependencies).

    Inputs:
    - ``sequence``: wild-type amino-acid string.
    - ``positions``: length-``L`` list of 1-indexed positions; ``"NC"`` entries get ``p=0.5``.
    - ``mutants``: mutant amino acid per site (list or dict). Shorter lists warn and
      pad with ``None`` (all non-wild-type amino acids). Required unless using ``"NC"`` only.
    - ``msa``: optional alignment path.
    - ``cache_folder``: HuggingFace cache directory for ESM weights.
    - ``ps_bounds``: ``(low, high)`` clip interval.

    Outputs:
    - ``(L,)`` float64 array of clipped mutation probabilities.
    """
    if not positions:
        raise ValueError("positions must be a non-empty list.")

    seq = validate_sequence(sequence)
    mutant_list = normalize_mutants(positions, mutants)

    if msa is not None:
        bounds = tuple(ps_bounds) if ps_bounds is not None else MSA_DEFAULT_BOUNDS
        ps = ps_from_msa(seq, positions, mutant_list, msa)
    else:
        bounds = tuple(ps_bounds) if ps_bounds is not None else ESM_DEFAULT_BOUNDS
        ps = ps_from_esm(seq, positions, mutant_list, cache_folder=cache_folder)

    return _clip_ps(ps, bounds).astype(np.float64)
