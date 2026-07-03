from __future__ import annotations

CANONICAL_AAS: frozenset[str] = frozenset("ACDEFGHIKLMNPQRSTVWY")
X_MUTANT_AAS: tuple[str, ...] = ("V", "L", "I")

MSA_DEFAULT_BOUNDS: tuple[float, float] = (0.05, 0.95)
ESM_DEFAULT_BOUNDS: tuple[float, float] = (0.1, 0.9)

ESM_MODEL_NAME: str = "facebook/esm2_t33_650M_UR50D"
