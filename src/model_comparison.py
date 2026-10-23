
from __future__ import annotations

import time
import warnings
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline

from src.models.pipeline import build_text_pipeline, build_url_pipeline

_MIN_SAMPLES_PER_FOLD = 1

_MIN_CALIBRATION_SAMPLES = 20

@dataclass
class ModelComparisonResult:

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
            "mean_roc_auc": (
                round(self.mean_roc_auc, 4)
                if not np.isnan(self.mean_roc_auc)
                else None
            ),
            "std_roc_auc": (
                round(self.std_roc_auc, 4)
                if not np.isnan(self.std_roc_auc)
                else None
            ),
            "fit_time_s":     round(self.fit_time_s, 3),
            "n_folds":        self.n_folds,
            "cv_warning":     self.cv_warning,
        }


def _safe_cv_folds(n_samples: int, n_classes: int, requested_cv: int) -> tuple[int, str]:

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

def _build_dummy_url_pipeline(random_state: int = 42) -> Any:

    from sklearn.feature_extraction import DictVectorizer

    return Pipeline([
        ("vec",   DictVectorizer(sparse=False)),
        ("clf",   DummyClassifier(strategy="stratified", random_state=random_state)),
    ])


def _build_lr_url_pipeline(random_state: int = 42) -> Any:
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



def _build_dummy_text_pipeline(random_state: int = 42) -> Any:
    from sklearn.feature_extraction.text import TfidfVectorizer

    return Pipeline([
        ("tfidf", TfidfVectorizer(max_features=500)),
        ("clf",   DummyClassifier(strategy="stratified", random_state=random_state)),
    ])


def _build_lr_text_pipeline(random_state: int = 42) -> Any:
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

def calibrate_pipeline(
    pipeline: Any,
    X_train: list,
    y_train: list[int],
    method: str = "sigmoid",
    cv: int = 3,
) -> Any:

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


def print_comparison_table(results: list[ModelComparisonResult]) -> None:

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
