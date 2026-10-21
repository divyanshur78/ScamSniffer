"""
src/models/pipeline.py
=======================
scikit-learn pipeline factories for ScamSniffer.

Two pipelines are provided:

  build_url_pipeline()
      Accepts a list of feature dicts produced by
      ``src.features.url_features.extract_url_features``.
      Uses DictVectorizer → StandardScaler → RandomForestClassifier.

  build_text_pipeline()
      Accepts a list of raw text strings (email / SMS / chat bodies).
      Uses TfidfVectorizer (word + char n-grams via FeatureUnion)
      → LogisticRegression.

Both pipelines:
  - Fit only on training data (no leakage by design — callers must split
    before calling ``fit``).
  - Support ``predict(X)`` returning integer labels {0, 1}.
  - Support ``predict_proba(X)`` returning class probabilities.
  - Accept a ``random_state`` parameter (default 42) for reproducibility.
  - Raise ``ValueError`` / ``TypeError`` on clearly malformed inputs.

Labels
------
  0 = legitimate / benign
  1 = phishing / scam

⚠️  These pipeline factories build *untrained* pipelines. Callers are
    responsible for supplying real labelled data and for evaluating
    performance before making any security decisions.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

FeatureDict = dict[str, int | float]
Label = int  # 0 or 1


# ---------------------------------------------------------------------------
# Input guards
# ---------------------------------------------------------------------------


def _validate_feature_dicts(X: Any) -> list[FeatureDict]:
    """Assert *X* is a non-empty list/tuple of dicts; return as list."""
    if not isinstance(X, (list, tuple)):
        raise TypeError(
            f"URL pipeline expects a list of feature dicts, got {type(X).__name__!r}."
        )
    if len(X) == 0:
        raise ValueError("URL pipeline received an empty feature list.")
    for i, item in enumerate(X):
        if not isinstance(item, dict):
            raise TypeError(
                f"Item at index {i} is {type(item).__name__!r}; expected a dict."
            )
    return list(X)


def _validate_texts(X: Any) -> list[str]:
    """Assert *X* is a non-empty list/tuple of strings; return as list."""
    if not isinstance(X, (list, tuple)):
        raise TypeError(
            f"Text pipeline expects a list of strings, got {type(X).__name__!r}."
        )
    if len(X) == 0:
        raise ValueError("Text pipeline received an empty input list.")
    for i, item in enumerate(X):
        if not isinstance(item, str):
            raise TypeError(
                f"Item at index {i} is {type(item).__name__!r}; expected a str."
            )
    return list(X)


def _validate_labels(y: Any, n: int) -> np.ndarray:
    """Assert *y* is array-like of {0, 1} with length *n*."""
    arr = np.asarray(y)
    if arr.ndim != 1:
        raise ValueError(f"Labels must be a 1-D array; got shape {arr.shape}.")
    if len(arr) != n:
        raise ValueError(
            f"Number of labels ({len(arr)}) does not match number of samples ({n})."
        )
    unique = set(arr.tolist())
    if not unique.issubset({0, 1}):
        raise ValueError(
            f"Labels must contain only 0 (legitimate) and 1 (phishing); "
            f"found: {sorted(unique)}."
        )
    return arr


# ---------------------------------------------------------------------------
# Thin sklearn-compatible wrappers that add input validation
# ---------------------------------------------------------------------------


class _UrlPipeline:
    """
    Wrapper around a sklearn Pipeline for URL feature dicts.

    Validates inputs before delegating to the underlying pipeline so that
    callers receive descriptive errors rather than cryptic numpy/sklearn ones.
    """

    def __init__(self, pipeline: Pipeline) -> None:
        self._pipeline = pipeline

    # --- sklearn-style fit / predict interface ---

    def fit(self, X: Any, y: Any) -> "_UrlPipeline":
        X_clean = _validate_feature_dicts(X)
        y_clean = _validate_labels(y, len(X_clean))
        self._pipeline.fit(X_clean, y_clean)
        return self

    def predict(self, X: Any) -> np.ndarray:
        X_clean = _validate_feature_dicts(X)
        return self._pipeline.predict(X_clean)

    def predict_proba(self, X: Any) -> np.ndarray:
        X_clean = _validate_feature_dicts(X)
        return self._pipeline.predict_proba(X_clean)

    # Expose the underlying pipeline for inspection / persistence.
    @property
    def pipeline(self) -> Pipeline:
        return self._pipeline

    # Allow attribute pass-through for sklearn compatibility checks.
    # Guard: during unpickling, __init__ has not yet run so _pipeline does not
    # exist yet.  Raising AttributeError instead of recursing tells pickle to
    # fall back to its default __setstate__ / __reduce__ behaviour.
    def __getattr__(self, name: str) -> Any:
        if name == "_pipeline":
            raise AttributeError(name)
        return getattr(self._pipeline, name)


class _TextPipeline:
    """
    Wrapper around a sklearn Pipeline for raw text strings.

    Validates inputs before delegating to the underlying pipeline.
    """

    def __init__(self, pipeline: Pipeline) -> None:
        self._pipeline = pipeline

    def fit(self, X: Any, y: Any) -> "_TextPipeline":
        X_clean = _validate_texts(X)
        y_clean = _validate_labels(y, len(X_clean))
        self._pipeline.fit(X_clean, y_clean)
        return self

    def predict(self, X: Any) -> np.ndarray:
        X_clean = _validate_texts(X)
        return self._pipeline.predict(X_clean)

    def predict_proba(self, X: Any) -> np.ndarray:
        X_clean = _validate_texts(X)
        return self._pipeline.predict_proba(X_clean)

    @property
    def pipeline(self) -> Pipeline:
        return self._pipeline

    def __getattr__(self, name: str) -> Any:
        if name == "_pipeline":
            raise AttributeError(name)
        return getattr(self._pipeline, name)


# ---------------------------------------------------------------------------
# Public factory functions
# ---------------------------------------------------------------------------


def build_url_pipeline(*, random_state: int = 42) -> _UrlPipeline:
    """
    Build an *untrained* URL-feature pipeline.

    Pipeline steps
    --------------
    1. ``DictVectorizer``       — converts feature dicts to a sparse matrix.
    2. ``StandardScaler``       — zero-mean / unit-variance normalisation
                                   (``with_mean=False`` for sparse compat).
    3. ``RandomForestClassifier`` — ensemble classifier; supports
                                    ``predict_proba``.

    Parameters
    ----------
    random_state:
        Seed for the RandomForestClassifier. Default 42.

    Returns
    -------
    _UrlPipeline
        Untrained wrapper; call ``.fit(X, y)`` before predicting.
    """
    sklearn_pipeline = Pipeline(
        steps=[
            ("vectorizer", DictVectorizer(sparse=False)),
            ("scaler", StandardScaler()),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=100,
                    max_depth=None,
                    random_state=random_state,
                    class_weight="balanced",
                ),
            ),
        ]
    )
    return _UrlPipeline(sklearn_pipeline)


def build_text_pipeline(*, random_state: int = 42) -> _TextPipeline:
    """
    Build an *untrained* text classification pipeline.

    Pipeline steps
    --------------
    1. ``FeatureUnion`` of:
       - Word-level TF-IDF (unigrams + bigrams, top 10 000 features)
       - Character-level TF-IDF (3–5 grams, top 5 000 features)
         catches obfuscated spellings (e.g. "p@ssw0rd")
    2. ``LogisticRegression``  — fast, calibrated probability estimates.

    Parameters
    ----------
    random_state:
        Seed for the LogisticRegression solver. Default 42.

    Returns
    -------
    _TextPipeline
        Untrained wrapper; call ``.fit(X, y)`` before predicting.
    """
    feature_union = FeatureUnion(
        transformer_list=[
            (
                "word_tfidf",
                TfidfVectorizer(
                    analyzer="word",
                    ngram_range=(1, 2),
                    max_features=10_000,
                    sublinear_tf=True,
                    strip_accents="unicode",
                    token_pattern=r"(?u)\b\w+\b",
                ),
            ),
            (
                "char_tfidf",
                TfidfVectorizer(
                    analyzer="char_wb",
                    ngram_range=(3, 5),
                    max_features=5_000,
                    sublinear_tf=True,
                ),
            ),
        ]
    )
    sklearn_pipeline = Pipeline(
        steps=[
            ("features", feature_union),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    random_state=random_state,
                    class_weight="balanced",
                    solver="lbfgs",
                ),
            ),
        ]
    )
    return _TextPipeline(sklearn_pipeline)
