"""
ewht: evolutionary Walsh-Hadamard transform for protein fitness landscapes.

This package provides fast eWHT / WHT transforms, evolutionary probability estimation,
data preprocessing helpers, and Lasso-based compressed sensing for combinatorial
deep mutational scanning experiments.
"""

from ewht.basis import ewht_basis, get_basis, restore_expectation_coeffs, wht_basis
from ewht.cs import CSResult, center_targets, run_cs_experiment, select_lambda_cv
from ewht.data import genotypes_from_dataframe, sample_evolutionary_sequences, subtract_mean
from ewht.examples import example_msa, example_ps_reference, load_example
from ewht.ewht import efwht, efwht_from_dataframe, get_stable_scaling_factors, iefwht, stable_to_expectation_coeffs
from ewht.ps import get_ps

__all__ = [
    "efwht",
    "efwht_from_dataframe",
    "iefwht",
    "get_stable_scaling_factors",
    "stable_to_expectation_coeffs",
    "get_basis",
    "ewht_basis",
    "wht_basis",
    "restore_expectation_coeffs",
    "run_cs_experiment",
    "select_lambda_cv",
    "center_targets",
    "CSResult",
    "get_ps",
    "genotypes_from_dataframe",
    "sample_evolutionary_sequences",
    "subtract_mean",
    "load_example",
    "example_msa",
    "example_ps_reference",
]

__version__ = "0.0.1"
