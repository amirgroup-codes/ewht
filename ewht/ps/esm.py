from __future__ import annotations

from pathlib import Path

import numpy as np

from ewht.ps._constants import CANONICAL_AAS, ESM_MODEL_NAME, X_MUTANT_AAS
from ewht.ps._parse import wt_aa_at


def _require_esm_deps():
    try:
        import torch  # noqa: F401
        from transformers import AutoTokenizer, EsmForMaskedLM  # noqa: F401
    except ImportError as exc:
        raise ImportError(
            "ESM ps computation requires optional dependencies. "
            "Install with: pip install 'ewht[esm]'"
        ) from exc


def _aa_token_id(tokenizer, aa: str) -> int:
    token_id = tokenizer.convert_tokens_to_ids(aa.upper())
    if token_id == tokenizer.unk_token_id:
        raise ValueError(f"Unknown amino acid token '{aa}' in ESM vocabulary.")
    return token_id


def _mutation_prob_from_logits(
    logits,
    tokenizer,
    wt_aa: str,
    mut_aa: str | None,
) -> float:
    import torch

    probs = torch.softmax(logits, dim=-1)
    wt_prob = probs[_aa_token_id(tokenizer, wt_aa)].item()

    if mut_aa is None:
        mut_prob = sum(
            probs[_aa_token_id(tokenizer, aa)].item()
            for aa in CANONICAL_AAS
            if aa != wt_aa.upper()
        )
    elif mut_aa.upper() == "X":
        mut_prob = sum(probs[_aa_token_id(tokenizer, aa)].item() for aa in X_MUTANT_AAS)
    else:
        mut_prob = probs[_aa_token_id(tokenizer, mut_aa)].item()

    total = wt_prob + mut_prob
    if total == 0.0:
        return 0.5
    return float(mut_prob / total)


def ps_from_esm(
    sequence: str,
    positions: list[int | str],
    mutant_list: list[str | None],
    *,
    cache_folder: str | Path | None = None,
    model_name: str = ESM_MODEL_NAME,
) -> np.ndarray:
    _require_esm_deps()
    import torch
    from transformers import AutoTokenizer, EsmForMaskedLM

    print("Computing ps from ESM2-650M")

    cache_kwargs = {}
    if cache_folder is not None:
        cache_kwargs["cache_dir"] = str(cache_folder)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(model_name, **cache_kwargs)
    model = EsmForMaskedLM.from_pretrained(model_name, **cache_kwargs).to(device)
    model.eval()

    tokens = tokenizer(sequence, return_tensors="pt")
    base_input_ids = tokens["input_ids"].to(device)

    ps: list[float] = []
    with torch.no_grad():
        for pos, mut_aa in zip(positions, mutant_list):
            if pos == "NC":
                ps.append(0.5)
                continue
            pos_i = int(pos)
            wt_aa = wt_aa_at(sequence, pos_i)
            masked_ids = base_input_ids.clone()
            masked_ids[0, pos_i] = tokenizer.mask_token_id
            pos_logits = model(masked_ids).logits[0, pos_i]
            ps.append(_mutation_prob_from_logits(pos_logits, tokenizer, wt_aa, mut_aa))

    return np.array(ps, dtype=np.float64)
