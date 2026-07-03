from __future__ import annotations

import pandas as pd


def resolve_genotype_column(df: pd.DataFrame) -> str:
    """Resolve a binary genotype column for CS input."""
    lower_exact = {col.lower(): col for col in df.columns}
    if "genotype" in lower_exact:
        return lower_exact["genotype"]
    for col in df.columns:
        lower = col.lower()
        if "genotype" in lower:
            return col
    for col in df.columns:
        lower = col.lower()
        if "sequence" in lower and "mutated" not in lower:
            return col
    raise KeyError(
        "Could not find genotype column. Expected a column named 'genotype' "
        f"or containing 'genotype'/'sequence'. Available: {list(df.columns)}"
    )


def _column_matches(candidate: str, col_lower: str) -> bool:
    """Match exact names or underscore-delimited tokens (not arbitrary substrings)."""
    if col_lower == candidate:
        return True
    return (
        col_lower.startswith(f"{candidate}_")
        or col_lower.endswith(f"_{candidate}")
        or f"_{candidate}_" in f"_{col_lower}_"
    )


def resolve_column(df: pd.DataFrame, candidates: list[str], *, purpose: str) -> str:
    """Resolve a column by case-insensitive token match with priority order."""
    if df.empty:
        raise ValueError(f"Cannot resolve {purpose} column on an empty DataFrame.")
    lower_map = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        cand = candidate.lower()
        for col_lower, col in lower_map.items():
            if _column_matches(cand, col_lower):
                return col
    raise KeyError(
        f"Could not find {purpose} column. Tried candidates {candidates}; "
        f"available columns: {list(df.columns)}"
    )
