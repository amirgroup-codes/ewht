from __future__ import annotations

import re
import warnings

import numpy as np
import pandas as pd

from ewht.ps._constants import CANONICAL_AAS, X_MUTANT_AAS


def parse_genotype_tokens(raw: object) -> list[int]:
    """Parse genotype tokens in {0,1} or {1,-1} encoding."""
    text = str(raw).strip()
    if not text:
        raise ValueError("Empty genotype string.")

    if re.fullmatch(r"[01]+", text):
        return [int(ch) for ch in text]

    tokens: list[int] = []
    i = 0
    while i < len(text):
        if text[i].isspace() or text[i] in ",|/":
            i += 1
            continue
        if text.startswith("-1", i):
            tokens.append(-1)
            i += 2
            continue
        if text[i] == "1":
            tokens.append(1)
            i += 1
            continue
        if text[i] == "0":
            tokens.append(0)
            i += 1
            continue
        raise ValueError(f"Invalid genotype token in {raw!r} at position {i}.")

    if not tokens:
        raise ValueError(f"Could not parse genotype from {raw!r}.")
    return tokens


def tokens_to_binary_string(tokens: list[int]) -> str:
    """Convert parsed tokens to an internal 0/1 genotype string."""
    if any(t not in (0, 1) for t in tokens):
        if any(t not in (1, -1) for t in tokens):
            raise ValueError(f"Genotype tokens must use 0/1 or 1/-1 encoding, got {tokens}.")
        return "".join("0" if t == 1 else "1" for t in tokens)
    return "".join(str(t) for t in tokens)


def normalize_genotype_string(raw: object, L: int | None = None) -> str:
    if isinstance(raw, (int, np.integer)):
        if L is None:
            raise ValueError("Genotype provided as integer requires length L.")
        return str(int(raw)).zfill(L)

    bits = tokens_to_binary_string(parse_genotype_tokens(raw))
    if L is not None and len(bits) != L:
        raise ValueError(f"Invalid genotype '{raw}': expected length {L}, got {len(bits)}.")
    return bits


def infer_genotype_length(series: pd.Series) -> int:
    """Infer landscape length L from a genotype column."""
    if pd.api.types.is_numeric_dtype(series):
        return max(len(str(int(v))) for v in series)
    max_len = 0
    for raw in series:
        bits = normalize_genotype_string(raw)
        max_len = max(max_len, len(bits))
    return max_len


def genotype_index(genotype: str) -> int:
    """Map a binary genotype string to lexicographic landscape index."""
    return int(genotype, 2)


def index_to_genotype(index: int, L: int) -> str:
    return format(index, f"0{L}b")


def validate_genotype_series(series: pd.Series, L: int | None = None) -> pd.Series:
    """Normalize and validate a genotype column."""
    if pd.api.types.is_numeric_dtype(series):
        series = series.map(lambda v: str(int(v)).zfill(L))
    else:
        series = series.astype(str).str.strip()
        series = series.map(lambda g: g.zfill(L) if set(g) <= {"0", "1"} and len(g) < L else g)
    normalized = series.map(lambda g: normalize_genotype_string(g, L=L))
    lengths = normalized.map(len).unique()
    if len(lengths) != 1:
        raise ValueError(f"All genotypes must have the same length; observed lengths {sorted(lengths)}.")
    if L is not None and lengths[0] != L:
        raise ValueError(f"Expected genotype length {L}, got {lengths[0]}.")
    if not all(set(g).issubset({"0", "1"}) for g in normalized):
        raise ValueError("Genotype column must contain only 0/1 after normalization.")
    return normalized


def mutant_matches(residue: str, mut_aa: str) -> bool:
    residue = residue.upper()
    mut_aa = mut_aa.upper()
    if mut_aa == "X":
        return residue in X_MUTANT_AAS
    return residue == mut_aa


def encode_genotype_from_sequence(
    sequence: str,
    positions: list[int | str],
    mutant_list: list[str | None],
    wt_sequence: str,
    *,
    mutant_label: str | None = None,
) -> str:
    bits: list[str] = []
    for pos, mut_aa in zip(positions, mutant_list):
        if pos == "NC":
            has_nc = bool(mutant_label and re.search(r"(^|;)NC($|;)", mutant_label))
            bits.append("1" if has_nc else "0")
            continue
        pos_i = int(pos)
        residue = sequence[pos_i - 1].upper()
        wt_aa = wt_sequence[pos_i - 1].upper()
        if mut_aa is None:
            raise ValueError(
                f"mutants must be specified for position {pos_i}; "
                "use an explicit genotype column if mapping is custom."
            )
        if residue == wt_aa:
            bits.append("0")
        elif mutant_matches(residue, mut_aa):
            bits.append("1")
        else:
            raise ValueError(
                f"Residue '{residue}' at position {pos_i} is neither wild-type '{wt_aa}' "
                f"nor mutant '{mut_aa}'."
            )
    return "".join(bits)


def build_landscape_weights(ps: np.ndarray) -> np.ndarray:
    weights = np.array([1.0], dtype=np.float64)
    for p in ps:
        weights = np.kron(weights, np.array([1.0 - p, p], dtype=np.float64))
    return weights
