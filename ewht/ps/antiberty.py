"""AntiBERTy pseudo-log-likelihood scoring for antibody landscapes.

Unlike the ESM2 wildtype-marginal score in ``ewht.ps.esm_marginal``, this is **not** a
site-additive score read off the wild-type sequence. Every variant is scored on its own
sequence with AntiBERTy's pseudo-log-likelihood, the protocol documented for the model
(https://github.com/jeffreyruffolo/AntiBERTy):

    PLL(s) = (1/L) sum_i log p(s_i | s_\\i)      one forward pass per masked position
    PPL(s) = exp(-PLL(s))                        pseudo-perplexity

Higher PLL means a more antibody-like sequence. Scores are used as returned by the
model (no fitness-direction flip).

``pseudo_log_likelihood`` reimplements ``antiberty.AntiBERTyRunner.pseudo_log_likelihood``
in batched form: the reference implementation scores one sequence at a time, which is far
too slow for libraries of tens of thousands of variants. The two agree to float precision.

Scoring a large library is expensive (roughly an hour for 65k variants on one GPU), so
cache the returned array and feed it back through ``ewht.score_landscape(..., scores=...)``
rather than rescoring.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Sequence

import numpy as np

# Sequences per forward pass are chosen to keep b * L * (L + 2)^2 roughly constant, since
# eager attention costs about (copies) x (L + 2)^2 per layer and copies = b * L.
BATCH_BUDGET = 90_000_000


def _require_antiberty_deps():
    try:
        import torch  # noqa: F401
        from antiberty.model.AntiBERTy import AntiBERTy  # noqa: F401
        from antiberty.tokenizer import AntiBERTyTokenizer  # noqa: F401
        from antiberty.utils.get_weights import get_weights  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "AntiBERTy scoring requires optional dependencies. "
            "Install with: pip install 'ewht[antiberty]'"
        ) from exc


@lru_cache(maxsize=2)
def load_antiberty(device: str | None = None):
    """
    Load AntiBERTy once and keep it cached.

    Inputs:
    - ``device``: torch device string; defaults to CUDA when available.

    Outputs:
    - ``(model, tokenizer, device)``. Weights come from the AntiBERTy package's own
      download location.
    """
    _require_antiberty_deps()
    import torch
    from antiberty.model.AntiBERTy import AntiBERTy
    from antiberty.tokenizer import AntiBERTyTokenizer
    from antiberty.utils.get_weights import get_weights

    path = get_weights()
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model = AntiBERTy.from_pretrained(path, attn_implementation="eager").eval().to(dev)
    return model, AntiBERTyTokenizer(), dev


def _token_ids(sequences: Sequence[str], tokenizer) -> np.ndarray:
    """(N, L) residue token ids; every sequence in a landscape shares one length."""
    lengths = {len(s) for s in sequences}
    if len(lengths) != 1:
        raise ValueError(
            f"AntiBERTy scoring expects one sequence length (substitutions only), "
            f"got lengths {sorted(lengths)}."
        )
    table = np.full(128, tokenizer.unk_token_id, dtype=np.int64)
    for token, idx in tokenizer.token_to_id.items():
        if len(token) == 1:
            table[ord(token)] = idx
    flat = np.frombuffer("".join(sequences).encode("ascii"), dtype=np.uint8)
    return table[flat].reshape(len(sequences), lengths.pop())


def auto_batch(L: int, budget: int = BATCH_BUDGET) -> int:
    """Sequences per forward pass for a landscape of sequence length ``L``."""
    return max(1, int(budget // (L * (L + 2) ** 2)))


def pseudo_log_likelihood(
    sequences: Sequence[str],
    *,
    device: str | None = None,
    batch_sequences: int | None = None,
    verbose: bool = False,
) -> np.ndarray:
    """
    AntiBERTy pseudo-log-likelihood of every sequence.

    Each position is masked in its own forward pass and the log-probability of the true
    residue is averaged over positions. Batched over (sequence, masked position) pairs.

    Inputs:
    - ``sequences``: length-``N`` sequence of equal-length amino-acid strings.
    - ``device``: torch device string; defaults to CUDA when available.
    - ``batch_sequences``: sequences per forward pass; chosen from the sequence length
      when not given.
    - ``verbose``: print progress.

    Outputs:
    - ``(N,)`` float64 array of PLLs, higher = more antibody-like.
    """
    _require_antiberty_deps()
    import torch

    sequences = [str(s).strip().upper() for s in sequences]
    if not sequences:
        raise ValueError("sequences must be non-empty.")
    model, tokenizer, dev = load_antiberty(device)
    ids = _token_ids(sequences, tokenizer)
    n, L = ids.shape
    step = batch_sequences or auto_batch(L)
    special = torch.tensor(tokenizer.all_special_ids, device=dev)
    pos = torch.arange(L, device=dev)

    out = np.empty(n, dtype=np.float64)
    for start in range(0, n, step):
        chunk = torch.as_tensor(ids[start:start + step], device=dev)      # b x L
        b = chunk.shape[0]
        # b*L masked copies: copy (i, j) is sequence i with position j masked.
        body = chunk[:, None, :].expand(b, L, L).clone()
        body[:, pos, pos] = tokenizer.mask_token_id
        tokens = torch.cat(
            [
                torch.full((b, L, 1), tokenizer.cls_token_id, device=dev),
                body,
                torch.full((b, L, 1), tokenizer.sep_token_id, device=dev),
            ],
            dim=2,
        ).reshape(b * L, L + 2)
        with torch.no_grad():
            logits = model(
                input_ids=tokens, attention_mask=torch.ones_like(tokens)
            ).prediction_logits
        # keep only the masked position of each copy: copy (i, j) -> token index j + 1
        logits = logits.reshape(b, L, L + 2, -1)[:, pos, pos + 1].float()
        logits[:, :, special] = -float("inf")
        log_probs = torch.log_softmax(logits, dim=-1).gather(2, chunk[:, :, None])[:, :, 0]
        out[start:start + b] = log_probs.mean(dim=1).double().cpu().numpy()
        if verbose:
            print(f"    AntiBERTy {min(start + b, n)}/{n}", flush=True)
    return out


def combine_readings(plls: np.ndarray) -> np.ndarray:
    """
    Combine the PLLs of several readings of one ambiguous library into one score.

    A library may leave a residue ambiguous (for example an ``X`` standing for any of
    V/L/I). Each reading is scored separately and the readings are averaged **on the
    pseudo-perplexity scale**, so a variant is scored by the mean perplexity of the
    sequences it could be. The result is returned as a PLL again, i.e.
    ``-log(mean_r exp(-PLL_r))``, which keeps the sign convention of this module.

    Inputs:
    - ``plls``: ``(R, N)`` float array of PLLs, one row per reading; a ``(N,)`` array is
      returned unchanged.

    Outputs:
    - ``(N,)`` float64 array.
    """
    plls = np.asarray(plls, dtype=float)
    if plls.ndim == 1:
        return plls
    if plls.ndim != 2:
        raise ValueError("plls must be 1D or 2D (readings x variants).")
    from scipy.special import logsumexp

    return -(logsumexp(-plls, axis=0) - np.log(plls.shape[0]))
