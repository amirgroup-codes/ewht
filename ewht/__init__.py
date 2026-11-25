"""
ewht: evolutionary Walsh-Hadamard transform for protein fitness landscapes.

This package provides fast eWHT / WHT transforms, evolutionary probability estimation,
data preprocessing helpers, and Lasso-based compressed sensing for combinatorial
deep mutational scanning experiments.
"""

from ewht.basis import ewht_basis, get_basis, restore_expectation_coeffs, wht_basis
from ewht.cs import (
    CSResult,
    center_targets,
    run_cs_experiment,
    select_lambda_cv,
    select_lambda_path,
)
from ewht.data import (
    genotypes_from_dataframe,
    sample_evolutionary_sequences,
    split_bin_train_test,
    subtract_mean,
)
from ewht.examples import example_msa, example_ps_reference, load_example
from ewht.ewht import efwht, efwht_from_dataframe, get_stable_scaling_factors, iefwht, stable_to_expectation_coeffs
from ewht.ps import (
    MODELS,
    PLM_BIN_COLUMN,
    PLM_SCORE_COLUMN,
    get_ps,
    get_ps_from_library,
    get_ps_from_scores,
    plm_bin,
    ps_from_bin,
    ps_from_scored_landscape,
    score_landscape,
    score_wt_marginal,
    site_llr,
    spearman_score_vs_fitness,
    variant_sequences,
    wt_llr_matrix,
)

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
    "select_lambda_path",
    "center_targets",
    "CSResult",
    "get_ps",
    "score_landscape",
    "get_ps_from_library",
    "get_ps_from_scores",
    "ps_from_scored_landscape",
    "score_wt_marginal",
    "wt_llr_matrix",
    "site_llr",
    "variant_sequences",
    "plm_bin",
    "ps_from_bin",
    "spearman_score_vs_fitness",
    "MODELS",
    "PLM_SCORE_COLUMN",
    "PLM_BIN_COLUMN",
    "genotypes_from_dataframe",
    "sample_evolutionary_sequences",
    "split_bin_train_test",
    "subtract_mean",
    "load_example",
    "example_msa",
    "example_ps_reference",
]

__version__ = "0.2.0"
