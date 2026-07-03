from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Lasso
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from ewht._io import SeqData, parse_seq_data
from ewht._ordering import reorder_coeffs
from ewht._validate import validate_ps
from ewht.basis import get_basis, restore_expectation_coeffs


DEFAULT_LAMBDAS = [m * 10**i for i in range(-5, 1) for m in (1, 5)]


@dataclass
class CSResult:
    best_lambda: float
    coeffs: np.ndarray
    intercept: float
    train_metrics: dict[str, float]
    test_metrics: dict[str, float]
    predictions: dict[str, np.ndarray]
    p_norms: np.ndarray | None
    basis: str
    cv_scores: dict[float, float]

    def to_dict(self) -> dict[str, Any]:
        return {
            "best_lambda": self.best_lambda,
            "coeffs": self.coeffs,
            "intercept": self.intercept,
            "train_metrics": self.train_metrics,
            "test_metrics": self.test_metrics,
            "predictions": self.predictions,
            "p_norms": self.p_norms,
            "basis": self.basis,
            "cv_scores": self.cv_scores,
        }


def ps_genotype_weights(bin_seqs: np.ndarray, ps: np.ndarray) -> np.ndarray:
    """
    Evolutionary Bernoulli weights for a set of sampled genotypes.

    Inputs:
    - ``bin_seqs``: ``(N, L)`` int8 array of binary genotypes.
    - ``ps``: ``(L,)`` float array of mutation probabilities.

    Outputs:
    - ``(N,)`` float array of normalized weights summing to 1.
    """
    ps = np.asarray(ps, dtype=float)
    weights = np.prod(np.where(bin_seqs == 1, ps, 1.0 - ps), axis=1).astype(float)
    weights = np.clip(weights, 0.0, None)
    if np.sum(weights) <= 0.0:
        return np.ones(len(weights), dtype=float) / len(weights)
    return weights / np.sum(weights)


def center_targets(
    y: np.ndarray,
    bin_seqs: np.ndarray,
    ps: np.ndarray,
    *,
    basis: str,
    center_value: float | None = None,
) -> tuple[np.ndarray, float]:
    """
    Center fitness values for eWHT or WHT analysis.

    For eWHT, subtracts the ps-weighted mean over sampled genotypes. For WHT,
    subtracts the uniform mean.

    Inputs:
    - ``y``: ``(N,)`` float fitness values aligned with ``bin_seqs`` rows.
    - ``bin_seqs``: ``(N, L)`` int8 binary genotypes.
    - ``ps``: ``(L,)`` float mutation probabilities.
    - ``basis``: ``{'eWHT', 'WHT'}``.
    - ``center_value``: optional float; if provided, subtract this instead of
      recomputing the mean (useful to apply the train-set center to test data).

    Outputs:
    - ``(N,)`` float centered fitness values.
    - ``center``: float scalar that was subtracted.
    """
    y = np.asarray(y, dtype=float)
    if center_value is not None:
        return y - center_value, center_value
    if basis.upper() == "WHT":
        center = float(np.mean(y))
    else:
        weights = ps_genotype_weights(bin_seqs, ps)
        center = float(np.sum(y * weights))
    return y - center, center


def select_lambda_cv(
    X: np.ndarray,
    y: np.ndarray,
    *,
    lambdas: list[float] | None = None,
    n_splits: int = 5,
    fit_intercept: bool = False,
    random_state: int = 42,
) -> tuple[float, dict[float, float]]:
    """
    Select Lasso regularization strength by cross-validation on training data.

    Inputs:
    - ``X``: ``(N, P)`` float design matrix.
    - ``y``: ``(N,)`` float centered fitness targets.
    - ``lambdas``: list of candidate ``alpha`` values for ``sklearn.linear_model.Lasso``.
    - ``n_splits``: number of CV folds (default 5).
    - ``fit_intercept``: bool passed to Lasso.
    - ``random_state``: int seed for fold shuffling.

    Outputs:
    - ``best_lambda``: float with highest mean CV R².
    - ``cv_scores``: dict mapping each lambda to its mean CV R².
    """
    if lambdas is None:
        lambdas = list(DEFAULT_LAMBDAS)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_scores: dict[float, list[float]] = {alpha: [] for alpha in lambdas}

    for train_idx, val_idx in kf.split(X):
        X_tr, X_val = X[train_idx], X[val_idx]
        y_tr, y_val = y[train_idx], y[val_idx]
        for alpha in lambdas:
            model = Lasso(alpha=alpha, fit_intercept=fit_intercept, max_iter=10000)
            model.fit(X_tr, y_tr)
            y_pred = model.predict(X_val)
            if len(y_val) <= 1 or np.all(y_val == y_val[0]):
                score = 0.0
            else:
                score = float(r2_score(y_val, y_pred))
            cv_scores[alpha].append(score)

    mean_scores = {alpha: float(np.mean(scores)) for alpha, scores in cv_scores.items()}
    best_lambda = max(mean_scores, key=mean_scores.get)
    return best_lambda, mean_scores


def run_cs_experiment(
    train: SeqData,
    test: SeqData,
    ps: np.ndarray,
    *,
    basis: str = "eWHT",
    center_by_ps: bool = True,
    lambdas: list[float] | None = None,
    n_cv_splits: int = 5,
    degree_order: bool = False,
    random_state: int = 42,
    return_predictions: bool = True,
) -> CSResult:
    """
    Run Lasso compressed sensing on an eWHT or WHT basis.

    Builds a stable-norm design matrix, centers targets, selects lambda by CV on
    train, and fits Lasso. Returned coefficients are in expectation norm and
    lexicographic order by default.

    Inputs:
    - ``train``: pandas DataFrame with ``genotype`` and fitness columns, or
      ``(bin_seqs, y)`` tuple with ``bin_seqs`` ``(N, L)`` int8 and ``y`` ``(N,)`` float.
    - ``test``: same format as ``train``.
    - ``ps``: ``(L,)`` float mutation probabilities.
    - ``basis``: ``{'eWHT', 'WHT'}``.
    - ``center_by_ps``: if ``True`` (default), ps-weighted centering on train and
      the same center applied to test; Lasso uses ``fit_intercept=False``.
    - ``lambdas``: optional list of Lasso ``alpha`` candidates for CV.
    - ``n_cv_splits``: number of CV folds on train.
    - ``degree_order``: if ``False`` (default), return coeffs in lex order.
    - ``random_state``: int seed for CV and reproducibility.

    Outputs:
    - ``CSResult`` dataclass with ``coeffs`` ``(2^L,)`` float, metrics, predictions,
      and the selected ``best_lambda``.
    """
    train_seqs, y_train = parse_seq_data(train)
    test_seqs, y_test = parse_seq_data(test)
    L = train_seqs.shape[1]
    if test_seqs.shape[1] != L:
        raise ValueError("Train and test sequences must have the same length L.")
    ps = validate_ps(ps, L)

    basis_key = basis.upper()
    if basis_key == "WHT":
        basis_norm = "expectation"
        p_norms = None
    else:
        basis_norm = "stable"
        basis_key = "EWHT"

    X_train, p_norms = _build_design(train_seqs, ps, basis_key=basis_key, norm=basis_norm)
    X_test, _ = _build_design(test_seqs, ps, basis_key=basis_key, norm=basis_norm)

    fit_intercept = not center_by_ps
    if center_by_ps:
        y_train_fit, center = center_targets(y_train, train_seqs, ps, basis=basis_key)
        y_test_fit, _ = center_targets(y_test, test_seqs, ps, basis=basis_key, center_value=center)
    else:
        center = 0.0
        y_train_fit = y_train.astype(float)
        y_test_fit = y_test.astype(float)

    best_lambda, cv_scores = select_lambda_cv(
        X_train,
        y_train_fit,
        lambdas=lambdas,
        n_splits=n_cv_splits,
        fit_intercept=fit_intercept,
        random_state=random_state,
    )

    model = Lasso(alpha=best_lambda, fit_intercept=fit_intercept, max_iter=10000)
    model.fit(X_train, y_train_fit)

    raw_coeffs = np.asarray(model.coef_, dtype=float)
    if p_norms is not None:
        coeffs = restore_expectation_coeffs(raw_coeffs, p_norms, degree_order=True)
    else:
        coeffs = raw_coeffs.copy()

    if not degree_order:
        coeffs = reorder_coeffs(coeffs, L, degree_order=False)

    intercept = float(model.intercept_) + center if fit_intercept else center

    y_train_pred = model.predict(X_train) + (0.0 if fit_intercept else center)
    y_test_pred = model.predict(X_test) + (0.0 if fit_intercept else center)

    train_metrics = _metrics(y_train, y_train_pred)
    test_metrics = _metrics(y_test, y_test_pred)

    predictions = {}
    if return_predictions:
        predictions = {"train": y_train_pred, "test": y_test_pred}

    return CSResult(
        best_lambda=best_lambda,
        coeffs=coeffs,
        intercept=intercept,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        predictions=predictions,
        p_norms=p_norms,
        basis=basis_key,
        cv_scores=cv_scores,
    )


def _build_design(
    bin_seqs: np.ndarray,
    ps: np.ndarray,
    *,
    basis_key: str,
    norm: str,
) -> tuple[np.ndarray, np.ndarray | None]:
    out = get_basis(bin_seqs, ps, basis=basis_key, norm=norm, degree_order=True)
    if isinstance(out, tuple):
        return out
    return out, None


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    if len(y_true) <= 1 or np.all(y_true == y_true[0]):
        rho = 0.0
        r2 = 0.0
    else:
        rho = float(spearmanr(y_true, y_pred).statistic)
        r2 = float(r2_score(y_true, y_pred))
    return {"r2": r2, "spearman": rho}
