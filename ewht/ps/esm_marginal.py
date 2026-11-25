"""ESM2 wildtype-marginal scoring: the zero-shot LLR scheme used in the paper.

One unmasked forward pass over the wild-type sequence gives a position x amino-acid
log-likelihood-ratio (LLR) matrix, and a variant's score is the sum of the LLRs of its
substitutions. The wild-type therefore scores 0 and the score is additive across sites,
so a whole combinatorial library costs a single forward pass.

Higher score = more plausible to the model. Scores are used as returned by the model
(no fitness-direction flip).

The tiling helpers for sequences longer than ESM2's 1022-residue window are adapted from
esm-variants, ``esm_variants_utils.py`` (MIT License, Copyright (c) 2023 Nadav Brandes and
Vasilis Ntranos).
"""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import pandas as pd

from ewht.ps._constants import CANONICAL_AAS, ESM_MODEL_NAME, X_MUTANT_AAS
from ewht.ps._parse import normalize_mutants, validate_sequence

AA_ORDER = [
    "K", "R", "H", "E", "D", "N", "Q", "T", "S", "C",
    "G", "A", "V", "L", "I", "M", "P", "Y", "F", "W",
]
ESM2_MAX_LEN = 1022


def _require_esm_deps():
    try:
        import torch  # noqa: F401
        from transformers import AutoTokenizer, EsmForMaskedLM  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "ESM2 scoring requires optional dependencies. "
            "Install with: pip install 'ewht[esm]'"
        ) from exc


def _chop(idx, min_overlap, max_len):
    return idx[max_len - min_overlap:-max_len + min_overlap]


def _intervals(idx, min_overlap=511, max_len=ESM2_MAX_LEN, parts=None):
    if parts is None:
        parts = []
    if len(idx) <= max_len:
        if parts[-2][-1] - parts[-1][0] < min_overlap:
            mid = idx[int(len(idx) / 2)]
            return parts + [np.arange(mid - int(max_len / 2), mid + int(max_len / 2))]
        return parts
    parts += [idx[:max_len], idx[-max_len:]]
    return _intervals(_chop(idx, min_overlap, max_len), min_overlap, max_len, parts=parts)


def _intervals_and_weights(seq_len, min_overlap=511, max_len=ESM2_MAX_LEN, s=16):
    """Overlapping windows covering the sequence, with sigmoid blending weights."""
    ints = _intervals(np.arange(seq_len), min_overlap=min_overlap, max_len=max_len)
    ints = [ints[i] for i in np.argsort([i[0] for i in ints])]
    a = int(np.round(min_overlap / 2))
    t = np.arange(max_len)
    f = np.ones(max_len)
    f[:a] = 1 / (1 + np.exp(-(t[:a] - a / 2) / s))
    f[max_len - a:] = 1 / (1 + np.exp((t[:a] - a / 2) / s))
    f0 = np.ones(max_len)
    f0[max_len - a:] = 1 / (1 + np.exp((t[:a] - a / 2) / s))
    fn = np.ones(max_len)
    fn[:a] = 1 / (1 + np.exp(-(t[:a] - a / 2) / s))
    filt = [f0] + [f for _ in ints[1:-1]] + [fn]
    M = np.zeros((len(ints), seq_len))
    for k, i in enumerate(ints):
        M[k, i] = filt[k]
    return ints, M / M.sum(0)


def _llr_frame(log_probs: np.ndarray, tokens: list[str], sequence: str) -> pd.DataFrame:
    """Rows = amino acids (AA_ORDER), columns = '{wt_aa} {pos}'; log p(aa) - log p(wt aa)."""
    frame = pd.DataFrame(log_probs, columns=tokens, index=list(sequence)).T
    frame = frame.loc[list(AA_ORDER)]
    frame.columns = [f"{aa} {i + 1}" for i, aa in enumerate(sequence)]
    wt_norm = np.diag(frame.loc[[c.split(" ")[0] for c in frame.columns]])
    return frame - wt_norm


def wt_llr_matrix(
    wt_sequence: str,
    *,
    cache_folder: str | Path | None = None,
    model_name: str = ESM_MODEL_NAME,
    device: str | None = None,
) -> pd.DataFrame:
    """
    Position x amino-acid log-likelihood-ratio matrix for a wild-type sequence.

    One unmasked forward pass of ESM2; sequences longer than 1022 residues are covered by
    overlapping windows blended with sigmoid weights.

    Inputs:
    - ``wt_sequence``: wild-type amino-acid string.
    - ``cache_folder``: HuggingFace cache directory for the model weights.
    - ``model_name``: masked-LM checkpoint; defaults to ESM2-650M.
    - ``device``: torch device string; defaults to CUDA when available.

    Outputs:
    - pandas DataFrame with the 20 canonical amino acids as rows and one column
      ``'{wt_aa} {pos}'`` per residue (1-indexed), holding ``log p(aa) - log p(wt)``.
    """
    _require_esm_deps()
    import torch
    from transformers import AutoTokenizer, EsmForMaskedLM

    seq = validate_sequence(wt_sequence)
    cache_kwargs = {"cache_dir": str(cache_folder)} if cache_folder is not None else {}
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    tokenizer = AutoTokenizer.from_pretrained(model_name, **cache_kwargs)
    model = EsmForMaskedLM.from_pretrained(model_name, **cache_kwargs).to(dev).eval()
    vocab = list(tokenizer.convert_ids_to_tokens(range(model.config.vocab_size)))

    def logits_for(window: str) -> np.ndarray:
        ids = tokenizer(window, return_tensors="pt")["input_ids"].to(dev)
        with torch.no_grad():
            out = torch.log_softmax(model(ids).logits, dim=-1)
        return out[0].cpu().numpy()[1:-1, :]

    if len(seq) <= ESM2_MAX_LEN:
        return _llr_frame(logits_for(seq), vocab, seq)

    ints, weights = _intervals_and_weights(len(seq), min_overlap=512, max_len=ESM2_MAX_LEN, s=20)
    full = np.zeros((len(seq), len(vocab)))
    for i, idx in enumerate(ints):
        window = "".join(np.asarray(list(seq))[idx])
        part = np.zeros((len(seq), len(vocab)))
        part[idx] = logits_for(window)
        full += (part.T * weights[i, :]).T
    return _llr_frame(full, vocab, seq)


def _llr_column(llr: pd.DataFrame, pos: int) -> str:
    for col in llr.columns:
        if int(col.split()[1]) == pos:
            return col
    raise KeyError(f"No LLR column for position {pos} in the wildtype matrix.")


def site_llr(llr: pd.DataFrame, pos: int, mut_aa: str | None) -> float:
    """
    LLR of one substitution.

    Inputs:
    - ``llr``: wildtype LLR matrix from ``wt_llr_matrix``.
    - ``pos``: 1-indexed position.
    - ``mut_aa``: mutant amino acid. ``'X'`` stands for any of V/L/I and ``None`` for any
      non-wild-type amino acid; both are combined with a log-sum-exp over the alternatives.

    Outputs:
    - float LLR.
    """
    col = _llr_column(llr, pos)
    wt_aa = col.split()[0].upper()
    if mut_aa is None:
        options = [aa for aa in AA_ORDER if aa != wt_aa]
    elif mut_aa.upper() == "X":
        options = list(X_MUTANT_AAS)
    else:
        mut = mut_aa.upper()
        if mut not in CANONICAL_AAS:
            raise ValueError(f"Invalid mutant amino acid '{mut_aa}'.")
        return float(llr.loc[mut, col])
    values = np.array([float(llr.loc[aa, col]) for aa in options], dtype=np.float64)
    return float(np.log(np.exp(values).sum()))


def score_wt_marginal(
    genotypes: Sequence[str],
    wt_sequence: str,
    positions: Sequence[int | str],
    mutants: Sequence[str | None] | Mapping[int | str, str] | None = None,
    *,
    llr: pd.DataFrame | None = None,
    cache_folder: str | Path | None = None,
    model_name: str = ESM_MODEL_NAME,
    device: str | None = None,
) -> np.ndarray:
    """
    Wildtype-marginal LLR score of every variant in a library.

    A variant's score is the sum of the per-site LLRs of its substitutions, so the
    wild-type scores 0. Sites given as ``"NC"`` (non-coding, not scorable by a protein
    language model) contribute 0.

    Inputs:
    - ``genotypes``: length-``N`` sequence of 0/1 genotype strings; bit ``j`` is site
      ``positions[j]``.
    - ``wt_sequence``: wild-type amino-acid string.
    - ``positions``: length-``L`` list of 1-indexed positions, or ``"NC"``.
    - ``mutants``: mutant amino acid per site (list or dict). ``None`` entries mean any
      non-wild-type amino acid.
    - ``llr``: optional precomputed matrix from ``wt_llr_matrix`` (skips the model run).
    - ``cache_folder``, ``model_name``, ``device``: passed to ``wt_llr_matrix``.

    Outputs:
    - ``(N,)`` float64 array of scores, higher = more plausible to the model.
    """
    seq = validate_sequence(wt_sequence)
    mutant_list = normalize_mutants(list(positions), mutants)
    if llr is None:
        llr = wt_llr_matrix(
            seq, cache_folder=cache_folder, model_name=model_name, device=device
        )

    per_site = np.array(
        [
            0.0 if pos == "NC" else site_llr(llr, int(pos), mut)
            for pos, mut in zip(positions, mutant_list)
        ],
        dtype=np.float64,
    )
    L = len(per_site)
    bits = np.stack(
        [np.fromiter(str(g).strip().zfill(L), dtype=np.int8) for g in genotypes]
    )
    if bits.shape[1] != L:
        raise ValueError(f"Genotypes have length {bits.shape[1]}, expected {L}.")
    return bits.astype(np.float64) @ per_site
