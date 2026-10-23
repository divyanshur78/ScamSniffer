
from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import StandardScaler

FeatureDict = dict[str, int | float]
Label = int


def _validate_feature_dicts(X: Any) -> list[FeatureDict]:
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


class _UrlPipeline:

    def __init__(self, pipeline: Pipeline) -> None:
        self._pipeline = pipeline

    def fit(self, X: Any, y: Any) -> _UrlPipeline:
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

    @property
    def pipeline(self) -> Pipeline:
        return self._pipeline

    def __getattr__(self, name: str) -> Any:
        if name == "_pipeline":
            raise AttributeError(name)
        return getattr(self._pipeline, name)


class _TextPipeline:

    def __init__(self, pipeline: Pipeline) -> None:
        self._pipeline = pipeline

    def fit(self, X: Any, y: Any) -> _TextPipeline:
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


def build_url_pipeline(*, random_state: int = 42) -> _UrlPipeline:

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
