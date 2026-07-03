from __future__ import annotations

import warnings
from typing import Mapping

from ewht.ps._constants import CANONICAL_AAS


def validate_sequence(sequence: str) -> str:
    seq = sequence.strip().upper()
    if not seq:
        raise ValueError("sequence must be a non-empty amino acid string.")
    invalid = {ch for ch in seq if ch not in CANONICAL_AAS}
    if invalid:
        raise ValueError(f"sequence contains non-canonical amino acids: {sorted(invalid)}")
    return seq


def normalize_mutants(
    positions: list[int | str],
    mutants: list[str | None] | Mapping[int | str, str] | None,
) -> list[str | None]:
    """Return one mutant specification per position (None = all non-WT)."""
    if mutants is None:
        return [None] * len(positions)

    if isinstance(mutants, Mapping):
        out: list[str | None] = []
        for pos in positions:
            if pos == "NC":
                out.append(None)
            elif pos in mutants:
                out.append(mutants[pos])
            elif int(pos) in mutants:  # type: ignore[arg-type]
                out.append(mutants[int(pos)])  # type: ignore[index]
            else:
                out.append(None)
        return out

    mutant_list = list(mutants)
    if len(mutant_list) < len(positions):
        warnings.warn(
            f"mutants length ({len(mutant_list)}) is shorter than positions ({len(positions)}); "
            "padding remaining entries with None (all non-wild-type amino acids).",
            stacklevel=3,
        )
        mutant_list = mutant_list + [None] * (len(positions) - len(mutant_list))
    elif len(mutant_list) > len(positions):
        raise ValueError(
            f"mutants length ({len(mutant_list)}) exceeds positions length ({len(positions)})."
        )
    return [
        None if pos == "NC" else mut
        for pos, mut in zip(positions, mutant_list)
    ]


def wt_aa_at(sequence: str, pos: int) -> str:
    if pos < 1 or pos > len(sequence):
        raise ValueError(f"Position {pos} is out of range for sequence of length {len(sequence)}.")
    return sequence[pos - 1]
