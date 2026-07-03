from __future__ import annotations

from pathlib import Path

import numpy as np

from ewht.ps._constants import CANONICAL_AAS, X_MUTANT_AAS
from ewht.ps._parse import wt_aa_at


def read_alignment(path: str | Path) -> list[str]:
    """Read a FASTA or A2M file into a list of alignment strings."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"MSA not found: {path}")

    sequences: list[str] = []
    current: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            if line.startswith(">"):
                if current:
                    sequences.append("".join(current))
                    current = []
            else:
                current.append(line)
    if current:
        sequences.append("".join(current))
    if not sequences:
        raise ValueError(f"No sequences found in MSA: {path}")
    return sequences


def _mutant_set(mut_aa: str | None, wt_aa: str) -> set[str] | None:
    if mut_aa is None:
        return None
    mut_aa = mut_aa.upper()
    if mut_aa == "X":
        return set(X_MUTANT_AAS)
    if mut_aa not in CANONICAL_AAS:
        raise ValueError(f"Invalid mutant amino acid '{mut_aa}'.")
    if mut_aa == wt_aa:
        raise ValueError(f"Mutant amino acid '{mut_aa}' matches wild-type at this position.")
    return {mut_aa}


def site_probability_from_msa(
    sequences: list[str],
    pos_1indexed: int,
    wt_aa: str,
    mut_aa: str | None,
) -> float:
    """
    Empirical p(mutant) from an alignment column.

    If mut_aa is None, mutant mass is any canonical non-WT amino acid.
    If mut_aa is 'X', mutant set is {V, L, I}.
    """
    wt_upper = wt_aa.upper()
    mut_set = _mutant_set(mut_aa, wt_upper)

    wt_count = 0
    mut_count = 0
    for seq in sequences:
        if pos_1indexed - 1 >= len(seq):
            continue
        ch = seq[pos_1indexed - 1]
        if not ch.isalpha():
            continue
        ch = ch.upper()
        if ch not in CANONICAL_AAS:
            continue
        if ch == wt_upper:
            wt_count += 1
        elif mut_set is None:
            mut_count += 1
        elif ch in mut_set:
            mut_count += 1

    total = wt_count + mut_count
    if total == 0:
        return 0.5
    return float(mut_count / total)


def ps_from_msa(
    sequence: str,
    positions: list[int | str],
    mutant_list: list[str | None],
    msa_path: str | Path,
) -> np.ndarray:
    sequences = read_alignment(msa_path)
    ps: list[float] = []
    for pos, mut_aa in zip(positions, mutant_list):
        if pos == "NC":
            ps.append(0.5)
            continue
        pos_i = int(pos)
        wt_aa = wt_aa_at(sequence, pos_i)
        ps.append(site_probability_from_msa(sequences, pos_i, wt_aa, mut_aa))
    return np.array(ps, dtype=np.float64)
