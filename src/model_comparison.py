"""
src/model_comparison.py
=======================
Multi-model comparison utilities for ScamSniffer.

Provides helpers to train and evaluate Dummy, Logistic Regression, and Random
Forest baselines in a reproducible way, with stratified cross-validation and
optional probability calibration.

⚠️  All metrics produced by this module on the bundled synthetic data are for
    development/comparison purposes ONLY and MUST NOT be cited as real-world
    performance figures.

Public API
----------
ModelComparisonResult
    Dataclass holding per-model cross-validation metrics and fit timing.

compare_url_models(feature_dicts, labels, cv, random_state)
    Train three URL-pipeline variants (Dummy, LR, RF) and return a list of
    ModelComparisonResult objects ranked by mean F1-score.

compare_text_models(texts, labels, cv, random_state)
    Train three text-pipeline variants and return ranked ModelComparisonResult
    objects.

calibrate_pipeline(pipeline, X_train, y_train, method, cv)
    Wrap a fitted pipeline in CalibratedClassifierCV (isotonic or sigmoid).
    Returns the calibrated pipeline.

print_comparison_table(results)
    Print a formatted comparison table to stdout.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.features.url_features import extract_url_features
from src.models.pipeline import build_text_pipeline, build_url_pipeline

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Minimum samples needed per class for StratifiedKFold with cv folds
_MIN_SAMPLES_PER_FOLD = 1

# Calibration minimum: need enough samples for the inner CV to work
_MIN_CALIBRATION_SAMPLES = 20


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class ModelComparisonResult:
    """
    Cross-validation metrics for a single model variant.

    Attributes
    ----------
    model_name     : str    Human-readable name of the model/variant.
    mean_precision : float  Mean precision across CV folds.
    std_precision  : float  Std deviation of precision across folds.
    mean_recall    : float  Mean recall across CV folds.
    std_recall     : float  Std deviation of recall across folds.
    mean_f1        : float  Mean F1-score across CV folds.
    std_f1         : float  Std deviation of F1 across folds.
    mean_roc_auc   : float  Mean ROC-AUC across CV folds (NaN if unavailable).
    std_roc_auc    : float  Std deviation of ROC-AUC.
    fit_time_s     : float  Wall-clock time to run cross_validate (seconds).
    n_folds        : int    Number of CV folds used.
    cv_warning     : str    Non-empty if cross-validation was degraded or
                            skipped due to insufficient data.
    """

    model_name:     str
    mean_precision: float = 0.0
    std_precision:  float = 0.0
    mean_recall:    float = 0.0
    std_recall:     float = 0.0
    mean_f1:        float = 0.0
    std_f1:         float = 0.0
    mean_roc_auc:   float = float("nan")
    std_roc_auc:    float = float("nan")
    fit_time_s:     float = 0.0
    n_folds:        int   = 0
    cv_warning:     str   = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "model_name":     self.model_name,
            "mean_precision": round(self.mean_precision, 4),
            "std_precision":  round(self.std_precision, 4),
            "mean_recall":    round(self.mean_recall, 4),
            "std_recall":     round(self.std_recall, 4),
            "mean_f1":        round(self.mean_f1, 4),
            "std_f1":         round(self.std_f1, 4),
            "mean_roc_auc":   round(self.mean_roc_auc, 4) if not np.isnan(self.mean_roc_auc) else None,
            "std_roc_auc":    round(self.std_roc_auc, 4) if not np.isnan(self.std_roc_auc) else None,
            "fit_time_s":     round(self.fit_time_s, 3),
            "n_folds":        self.n_folds,
            "cv_warning":     self.cv_warning,
        }


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _safe_cv_folds(n_samples: int, n_classes: int, requested_cv: int) -> tuple[int, str]:
    """
    Return (actual_folds, warning_message).

    Reduces *requested_cv* to the maximum value that still leaves at least
    *_MIN_SAMPLES_PER_FOLD* sample per class per fold.  Returns (0, reason)
    when no valid CV is possible.
    """
    min_per_class = n_samples // n_classes
    max_folds = min(min_per_class, n_samples)
    actual = min(requested_cv, max_folds)
    if actual < 2:
        return 0, (
            f"Cannot run cross-validation: only {min_per_class} sample(s) per class "
            f"with {n_samples} total — need ≥2 per class per fold."
        )
    warning = ""
    if actual < requested_cv:
        warning = (
            f"Requested cv={requested_cv} reduced to cv={actual} due to small "
            f"dataset ({n_samples} samples, {n_classes} classes)."
        )
    return actual, warning


def _run_cv(
    pipeline: Any,
    X: list,
    y: list[int],
    cv: int,
    random_state: int,
    model_name: str,
) -> ModelComparisonResult:
    """
    Run stratified k-fold CV and return a ModelComparisonResult.

    If the dataset is too small for CV, returns zero-filled metrics with
    a descriptive warning.
    """
    n_classes = len(set(y))
    actual_cv, cv_warning = _safe_cv_folds(len(y), n_classes, cv)

    result = ModelComparisonResult(
        model_name=model_name,
        cv_warning=cv_warning,
    )

    if actual_cv < 2:
        result.n_folds = 0
        return result

    skf = StratifiedKFold(n_splits=actual_cv, shuffle=True, random_state=random_state)
    scoring = {
        "precision": "precision",
        "recall":    "recall",
        "f1":        "f1",
        "roc_auc":   "roc_auc",
    }

    t0 = time.perf_counter()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv_scores = cross_validate(
            pipeline, X, y, cv=skf, scoring=scoring,
            error_score=0.0,
        )
    elapsed = time.perf_counter() - t0

    result.mean_precision = float(np.mean(cv_scores["test_precision"]))
    result.std_precision  = float(np.std(cv_scores["test_precision"]))
    result.mean_recall    = float(np.mean(cv_scores["test_recall"]))
    result.std_recall     = float(np.std(cv_scores["test_recall"]))
    result.mean_f1        = float(np.mean(cv_scores["test_f1"]))
    result.std_f1         = float(np.std(cv_scores["test_f1"]))
    result.mean_roc_auc   = float(np.mean(cv_scores["test_roc_auc"]))
    result.std_roc_auc    = float(np.std(cv_scores["test_roc_auc"]))
    result.fit_time_s     = elapsed
    result.n_folds        = actual_cv

    return result


# ---------------------------------------------------------------------------
# URL model comparison
# ---------------------------------------------------------------------------


def _build_dummy_url_pipeline(random_state: int = 42) -> Any:
    """
    Build a Dummy URL pipeline (stratified random predictions).

    Wraps DummyClassifier in the same dict-vectoriser scaffold as the real
    pipeline so it accepts a list[dict] input.
    """
    from sklearn.feature_extraction import DictVectorizer

    return Pipeline([
        ("vec",   DictVectorizer(sparse=False)),
        ("clf",   DummyClassifier(strategy="stratified", random_state=random_state)),
    ])


def _build_lr_url_pipeline(random_state: int = 42) -> Any:
    """Build a Logistic Regression URL pipeline."""
    from sklearn.feature_extraction import DictVectorizer
    from sklearn.preprocessing import StandardScaler

    return Pipeline([
        ("vec",    DictVectorizer(sparse=False)),
        ("scaler", StandardScaler()),
        ("clf",    LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            class_weight="balanced",
        )),
    ])


def compare_url_models(
    feature_dicts: list[dict],
    labels: list[int],
    cv: int = 5,
    random_state: int = 42,
) -> list[ModelComparisonResult]:
    """
    Train and cross-validate three URL-classification pipeline variants.

    Variants
    --------
    1. DummyClassifier   — stratified random baseline
    2. LogisticRegression — linear baseline with standard scaling
    3. RandomForest       — the default ScamSniffer pipeline

    Parameters
    ----------
    feature_dicts : list of feature dicts from extract_url_features()
    labels        : corresponding integer labels (0 or 1)
    cv            : number of CV folds (automatically reduced for tiny datasets)
    random_state  : random seed for reproducibility

    Returns
    -------
    List of ModelComparisonResult objects, sorted descending by mean_f1.
    """
    pipelines = [
        ("Dummy (stratified)",     _build_dummy_url_pipeline(random_state)),
        ("LogisticRegression",     _build_lr_url_pipeline(random_state)),
        ("RandomForest (default)", build_url_pipeline(random_state=random_state)),
    ]

    results = []
    for name, pipe in pipelines:
        result = _run_cv(pipe, feature_dicts, labels, cv, random_state, name)
        results.append(result)

    results.sort(key=lambda r: r.mean_f1, reverse=True)
    return results


# ---------------------------------------------------------------------------
# Text model comparison
# ---------------------------------------------------------------------------


def _build_dummy_text_pipeline(random_state: int = 42) -> Any:
    """Build a Dummy text pipeline."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=500)),
        ("clf",   DummyClassifier(strategy="stratified", random_state=random_state)),
    ])


def _build_lr_text_pipeline(random_state: int = 42) -> Any:
    """Build a Logistic Regression text pipeline."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=5000)),
        ("clf",   LogisticRegression(
            max_iter=1000,
            random_state=random_state,
            class_weight="balanced",
        )),
    ])


def compare_text_models(
    texts: list[str],
    labels: list[int],
    cv: int = 5,
    random_state: int = 42,
) -> list[ModelComparisonResult]:
    """
    Train and cross-validate three text-classification pipeline variants.

    Variants
    --------
    1. DummyClassifier   — stratified random baseline
    2. LogisticRegression — TF-IDF + LR
    3. RandomForest       — the default ScamSniffer text pipeline

    Parameters
    ----------
    texts        : list of raw text strings
    labels       : corresponding integer labels (0 or 1)
    cv           : number of CV folds
    random_state : random seed

    Returns
    -------
    List of ModelComparisonResult objects, sorted descending by mean_f1.
    """
    pipelines = [
        ("Dummy (stratified)",     _build_dummy_text_pipeline(random_state)),
        ("LogisticRegression",     _build_lr_text_pipeline(random_state)),
        ("RandomForest (default)", build_text_pipeline(random_state=random_state)),
    ]

    results = []
    for name, pipe in pipelines:
        result = _run_cv(pipe, texts, labels, cv, random_state, name)
        results.append(result)

    results.sort(key=lambda r: r.mean_f1, reverse=True)
    return results


# ---------------------------------------------------------------------------
# Calibration
# ---------------------------------------------------------------------------


def calibrate_pipeline(
    pipeline: Any,
    X_train: list,
    y_train: list[int],
    method: str = "sigmoid",
    cv: int = 3,
) -> Any:
    """
    Wrap a fitted pipeline with probability calibration.

    ⚠️  Calibration is only meaningful with sufficient INDEPENDENT data.
        Do NOT calibrate and evaluate on the same samples.  With synthetic
        data of ≤120 samples the calibrated probabilities are NOT trustworthy
        as real confidence scores.

    Parameters
    ----------
    pipeline : a fitted scikit-learn pipeline (url or text)
    X_train  : training features (list[dict] for URL, list[str] for text)
    y_train  : training labels
    method   : ``"sigmoid"`` (Platt scaling) or ``"isotonic"``
    cv       : number of inner CV folds for calibration; use ``"prefit"``
               if pipeline is already fitted and you want to calibrate on
               held-out data only (pass calibration set as X_train/y_train).

    Returns
    -------
    A CalibratedClassifierCV fitted on (X_train, y_train).

    Raises
    ------
    ValueError
        If the training set is too small for reliable calibration
        (< _MIN_CALIBRATION_SAMPLES).
    """
    n = len(y_train)
    if n < _MIN_CALIBRATION_SAMPLES:
        raise ValueError(
            f"Calibration requires at least {_MIN_CALIBRATION_SAMPLES} independent "
            f"samples; got {n}.  ⚠️  Calibrated probabilities on synthetic data "
            "MUST NOT be interpreted as reliable confidence scores."
        )

    calibrated = CalibratedClassifierCV(
        estimator=pipeline, method=method, cv=cv
    )
    calibrated.fit(X_train, y_train)
    return calibrated


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------


def print_comparison_table(results: list[ModelComparisonResult]) -> None:
    """
    Print a formatted comparison table to stdout.

    Each row shows model name, mean±std for precision/recall/F1/ROC-AUC,
    and wall-clock fit time.
    """
    header = (
        f"{'Model':<35} {'Prec':>7} {'Rec':>7} {'F1':>7} {'AUC':>7} {'Time(s)':>8}"
    )
    sep = "-" * len(header)
    print(sep)
    print(header)
    print(sep)
    for r in results:
        auc_str = (
            f"{r.mean_roc_auc:.3f}" if not np.isnan(r.mean_roc_auc) else "  N/A "
        )
        print(
            f"{r.model_name:<35} "
            f"{r.mean_precision:.3f}±{r.std_precision:.2f}  "
            f"{r.mean_recall:.3f}±{r.std_recall:.2f}  "
            f"{r.mean_f1:.3f}±{r.std_f1:.2f}  "
            f"{auc_str}  "
            f"{r.fit_time_s:>7.2f}s"
        )
        if r.cv_warning:
            print(f"  ⚠  {r.cv_warning}")
    print(sep)
    print("⚠️  All metrics are from SYNTHETIC data and must not be cited as real-world accuracy.")
