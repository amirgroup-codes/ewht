"""Bundled example datasets for tutorials and demos."""

from __future__ import annotations

from contextlib import contextmanager
from importlib.resources import as_file, files
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

CR6261_WT_SEQUENCE = (
    "QVQLVESGAEVKKPGSSVKVSCKASGGTFSSYAISWVRQAPGQGPEWMGGIIPIFGTANYAQKFQGRVTITADKSTSTAYMELSSLRSEDTAMYYCAKHMGYQLRETMDVWGQGTTVTVSS"
)
CR6261_POSITIONS = [28, 30, 58, 59, 62, 74, 75, 76, 77, 79, 104]
CR6261_MUTANTS = ["P", "R", "T", "K", "P", "D", "F", "A", "G", "V", "V"]

_EXAMPLE_CSV = "example_data/cr6261_h1_mean.csv"
_EXAMPLE_MSA = "example_data/cr6261.a2m"
_EXAMPLE_PS = "example_data/cr6261_ps.npy"


def load_example() -> pd.DataFrame:
    """
    Load the bundled CR6261 H1-mean deep mutational scanning dataset.

    The CSV contains ``mutant``, ``mutated_sequence``, ``fitness``, and
    ``estimated_fitness`` columns without precomputed genotype fields.

    Outputs:
    - pandas DataFrame with 2048 rows (full ``2^11`` combinatorial library).
    """
    path = files("ewht").joinpath(_EXAMPLE_CSV)
    return pd.read_csv(path)


@contextmanager
def example_msa() -> Iterator[Path]:
    """
    Yield a filesystem path to the bundled CR6261 MSA (``.a2m``).

    Outputs:
    - ``pathlib.Path`` to a temporary or source-tree copy of the alignment file.
    """
    with as_file(files("ewht").joinpath(_EXAMPLE_MSA)) as path:
        yield Path(path)


def example_ps_reference() -> np.ndarray:
    """
    Load reference ps values bundled with the example dataset.

    Outputs:
    - ``(11,)`` float64 array of MSA-derived mutation probabilities.
    """
    with as_file(files("ewht").joinpath(_EXAMPLE_PS)) as path:
        return np.load(path)
