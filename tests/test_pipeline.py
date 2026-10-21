"""
tests/test_pipeline.py
=======================
Unit tests for src/models/pipeline.py.

Coverage targets:
  - Pipeline construction (factory functions return correct types)
  - URL pipeline: fit + predict on tiny in-memory feature-dict examples
  - URL pipeline: predict_proba shape and probability bounds
  - URL pipeline: determinism via random_state
  - URL pipeline: custom random_state accepted
  - URL pipeline: invalid inputs (wrong type, empty list, bad labels)
  - Text pipeline: fit + predict on tiny in-memory string examples
  - Text pipeline: predict_proba shape and probability bounds
  - Text pipeline: determinism via random_state
  - Text pipeline: invalid inputs (wrong type, empty list, bad labels)
  - Label validation: wrong length, out-of-range values, 2-D array
  - Pipeline independence: URL and text pipelines are separate objects

All training data is tiny and synthetic — no real dataset is loaded.
No model metrics are asserted; only structural correctness is verified.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from src.models.pipeline import (
    _validate_feature_dicts,
    _validate_labels,
    _validate_texts,
    build_text_pipeline,
    build_url_pipeline,
)

# ---------------------------------------------------------------------------
# Minimal synthetic training data
# ---------------------------------------------------------------------------

# Two toy URL feature dicts — just enough keys for DictVectorizer to work.
URL_FEATURES_LEGIT: list[dict] = [
    {"url_length": 25, "is_https": 1, "suspicious_kw_count": 0,
     "has_suspicious_tld": 0, "domain_entropy": 2.5, "is_ip_host": 0,
     "has_non_std_port": 0, "dot_count": 2, "subdomain_depth": 1,
     "hostname_length": 11, "path_length": 5, "query_length": 0},
    {"url_length": 30, "is_https": 1, "suspicious_kw_count": 0,
     "has_suspicious_tld": 0, "domain_entropy": 2.8, "is_ip_host": 0,
     "has_non_std_port": 0, "dot_count": 2, "subdomain_depth": 0,
     "hostname_length": 14, "path_length": 7, "query_length": 3},
    {"url_length": 22, "is_https": 1, "suspicious_kw_count": 0,
     "has_suspicious_tld": 0, "domain_entropy": 2.1, "is_ip_host": 0,
     "has_non_std_port": 0, "dot_count": 1, "subdomain_depth": 0,
     "hostname_length": 10, "path_length": 4, "query_length": 0},
]

URL_FEATURES_PHISH: list[dict] = [
    {"url_length": 95, "is_https": 0, "suspicious_kw_count": 5,
     "has_suspicious_tld": 1, "domain_entropy": 3.9, "is_ip_host": 1,
     "has_non_std_port": 1, "dot_count": 6, "subdomain_depth": 3,
     "hostname_length": 11, "path_length": 40, "query_length": 20},
    {"url_length": 110, "is_https": 0, "suspicious_kw_count": 6,
     "has_suspicious_tld": 1, "domain_entropy": 4.1, "is_ip_host": 0,
     "has_non_std_port": 1, "dot_count": 8, "subdomain_depth": 4,
     "hostname_length": 30, "path_length": 50, "query_length": 30},
    {"url_length": 88, "is_https": 0, "suspicious_kw_count": 4,
     "has_suspicious_tld": 1, "domain_entropy": 3.7, "is_ip_host": 0,
     "has_non_std_port": 0, "dot_count": 5, "subdomain_depth": 2,
     "hostname_length": 25, "path_length": 35, "query_length": 15},
]

URL_X_TRAIN: list[dict] = URL_FEATURES_LEGIT + URL_FEATURES_PHISH
URL_Y_TRAIN: list[int] = [0, 0, 0, 1, 1, 1]

# Tiny text corpus.
TEXT_X_TRAIN: list[str] = [
    "Hi there, just checking in how you are doing.",
    "Meeting rescheduled to Thursday afternoon.",
    "Your package has been delivered successfully.",
    "URGENT: Your account has been suspended! Verify password immediately!",
    "Click here to claim your prize: http://phishing.xyz/login",
    "Bank account locked. Confirm credentials now or lose access forever.",
]
TEXT_Y_TRAIN: list[int] = [0, 0, 0, 1, 1, 1]


# ===========================================================================
# Input validation helpers
# ===========================================================================


class TestValidateFeatureDicts:
    def test_valid_list_of_dicts_returns_list(self):
        result = _validate_feature_dicts([{"a": 1}, {"a": 2}])
        assert result == [{"a": 1}, {"a": 2}]

    def test_tuple_accepted(self):
        result = _validate_feature_dicts(({"a": 1},))
        assert isinstance(result, list)

    def test_empty_list_raises_valueerror(self):
        with pytest.raises(ValueError, match="empty"):
            _validate_feature_dicts([])

    def test_non_list_raises_typeerror(self):
        with pytest.raises(TypeError, match="list"):
            _validate_feature_dicts({"a": 1})

    def test_non_dict_item_raises_typeerror(self):
        with pytest.raises(TypeError, match="index 1"):
            _validate_feature_dicts([{"a": 1}, "not a dict"])


class TestValidateTexts:
    def test_valid_list_of_strings_returns_list(self):
        result = _validate_texts(["hello", "world"])
        assert result == ["hello", "world"]

    def test_tuple_accepted(self):
        result = _validate_texts(("hello",))
        assert isinstance(result, list)

    def test_empty_list_raises_valueerror(self):
        with pytest.raises(ValueError, match="empty"):
            _validate_texts([])

    def test_non_list_raises_typeerror(self):
        with pytest.raises(TypeError, match="list"):
            _validate_texts("a plain string")

    def test_non_string_item_raises_typeerror(self):
        with pytest.raises(TypeError, match="index 0"):
            _validate_texts([42])


class TestValidateLabels:
    def test_valid_binary_labels(self):
        arr = _validate_labels([0, 1, 0, 1], 4)
        assert list(arr) == [0, 1, 0, 1]

    def test_length_mismatch_raises_valueerror(self):
        with pytest.raises(ValueError, match="does not match"):
            _validate_labels([0, 1], 3)

    def test_out_of_range_label_raises_valueerror(self):
        with pytest.raises(ValueError, match="0.*1"):
            _validate_labels([0, 2], 2)

    def test_two_d_array_raises_valueerror(self):
        with pytest.raises(ValueError, match="1-D"):
            _validate_labels([[0, 1], [1, 0]], 2)

    def test_all_zeros_accepted(self):
        arr = _validate_labels([0, 0, 0], 3)
        assert list(arr) == [0, 0, 0]

    def test_all_ones_accepted(self):
        arr = _validate_labels([1, 1], 2)
        assert list(arr) == [1, 1]


# ===========================================================================
# URL pipeline — construction
# ===========================================================================


class TestBuildUrlPipeline:
    def test_returns_url_pipeline_type(self):
        from src.models.pipeline import _UrlPipeline
        p = build_url_pipeline()
        assert isinstance(p, _UrlPipeline)

    def test_inner_pipeline_is_sklearn_pipeline(self):
        p = build_url_pipeline()
        assert isinstance(p.pipeline, Pipeline)

    def test_pipeline_has_three_steps(self):
        p = build_url_pipeline()
        assert len(p.pipeline.steps) == 3

    def test_step_names(self):
        p = build_url_pipeline()
        names = [name for name, _ in p.pipeline.steps]
        assert names == ["vectorizer", "scaler", "classifier"]

    def test_custom_random_state_accepted(self):
        p = build_url_pipeline(random_state=0)
        assert p.pipeline.named_steps["classifier"].random_state == 0

    def test_default_random_state_is_42(self):
        p = build_url_pipeline()
        assert p.pipeline.named_steps["classifier"].random_state == 42


# ===========================================================================
# URL pipeline — fit / predict
# ===========================================================================


class TestUrlPipelineFitPredict:
    def setup_method(self):
        self.pipeline = build_url_pipeline(random_state=42)
        self.pipeline.fit(URL_X_TRAIN, URL_Y_TRAIN)

    def test_predict_returns_array(self):
        preds = self.pipeline.predict(URL_X_TRAIN)
        assert isinstance(preds, np.ndarray)

    def test_predict_length_matches_input(self):
        preds = self.pipeline.predict(URL_X_TRAIN)
        assert len(preds) == len(URL_X_TRAIN)

    def test_predict_values_are_binary(self):
        preds = self.pipeline.predict(URL_X_TRAIN)
        assert set(preds.tolist()).issubset({0, 1})

    def test_predict_proba_shape(self):
        proba = self.pipeline.predict_proba(URL_X_TRAIN)
        assert proba.shape == (len(URL_X_TRAIN), 2)

    def test_predict_proba_rows_sum_to_one(self):
        proba = self.pipeline.predict_proba(URL_X_TRAIN)
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_predict_proba_values_in_0_1(self):
        proba = self.pipeline.predict_proba(URL_X_TRAIN)
        assert (proba >= 0.0).all() and (proba <= 1.0).all()

    def test_determinism_same_random_state(self):
        p1 = build_url_pipeline(random_state=7)
        p2 = build_url_pipeline(random_state=7)
        p1.fit(URL_X_TRAIN, URL_Y_TRAIN)
        p2.fit(URL_X_TRAIN, URL_Y_TRAIN)
        np.testing.assert_array_equal(
            p1.predict(URL_X_TRAIN),
            p2.predict(URL_X_TRAIN),
        )


# ===========================================================================
# URL pipeline — invalid inputs at predict time
# ===========================================================================


class TestUrlPipelineInvalidPredict:
    def setup_method(self):
        self.pipeline = build_url_pipeline()
        self.pipeline.fit(URL_X_TRAIN, URL_Y_TRAIN)

    def test_non_list_raises_typeerror(self):
        with pytest.raises(TypeError):
            self.pipeline.predict(URL_X_TRAIN[0])  # single dict, not list

    def test_empty_list_raises_valueerror(self):
        with pytest.raises(ValueError, match="empty"):
            self.pipeline.predict([])

    def test_list_of_strings_raises_typeerror(self):
        with pytest.raises(TypeError):
            self.pipeline.predict(["not a dict"])


# ===========================================================================
# URL pipeline — invalid inputs at fit time
# ===========================================================================


class TestUrlPipelineInvalidFit:
    def test_wrong_label_length_raises_valueerror(self):
        p = build_url_pipeline()
        with pytest.raises(ValueError, match="does not match"):
            p.fit(URL_X_TRAIN, [0, 1])  # wrong number of labels

    def test_out_of_range_label_raises_valueerror(self):
        p = build_url_pipeline()
        bad_y = [0, 0, 0, 1, 1, 2]  # label 2 is invalid
        with pytest.raises(ValueError, match="0.*1"):
            p.fit(URL_X_TRAIN, bad_y)

    def test_non_list_X_raises_typeerror(self):
        p = build_url_pipeline()
        with pytest.raises(TypeError):
            p.fit("not a list", URL_Y_TRAIN)


# ===========================================================================
# Text pipeline — construction
# ===========================================================================


class TestBuildTextPipeline:
    def test_returns_text_pipeline_type(self):
        from src.models.pipeline import _TextPipeline
        p = build_text_pipeline()
        assert isinstance(p, _TextPipeline)

    def test_inner_pipeline_is_sklearn_pipeline(self):
        p = build_text_pipeline()
        assert isinstance(p.pipeline, Pipeline)

    def test_pipeline_has_two_steps(self):
        p = build_text_pipeline()
        assert len(p.pipeline.steps) == 2

    def test_step_names(self):
        p = build_text_pipeline()
        names = [name for name, _ in p.pipeline.steps]
        assert names == ["features", "classifier"]

    def test_custom_random_state_accepted(self):
        p = build_text_pipeline(random_state=99)
        assert p.pipeline.named_steps["classifier"].random_state == 99

    def test_default_random_state_is_42(self):
        p = build_text_pipeline()
        assert p.pipeline.named_steps["classifier"].random_state == 42


# ===========================================================================
# Text pipeline — fit / predict
# ===========================================================================


class TestTextPipelineFitPredict:
    def setup_method(self):
        self.pipeline = build_text_pipeline(random_state=42)
        self.pipeline.fit(TEXT_X_TRAIN, TEXT_Y_TRAIN)

    def test_predict_returns_array(self):
        preds = self.pipeline.predict(TEXT_X_TRAIN)
        assert isinstance(preds, np.ndarray)

    def test_predict_length_matches_input(self):
        preds = self.pipeline.predict(TEXT_X_TRAIN)
        assert len(preds) == len(TEXT_X_TRAIN)

    def test_predict_values_are_binary(self):
        preds = self.pipeline.predict(TEXT_X_TRAIN)
        assert set(preds.tolist()).issubset({0, 1})

    def test_predict_proba_shape(self):
        proba = self.pipeline.predict_proba(TEXT_X_TRAIN)
        assert proba.shape == (len(TEXT_X_TRAIN), 2)

    def test_predict_proba_rows_sum_to_one(self):
        proba = self.pipeline.predict_proba(TEXT_X_TRAIN)
        np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)

    def test_predict_proba_values_in_0_1(self):
        proba = self.pipeline.predict_proba(TEXT_X_TRAIN)
        assert (proba >= 0.0).all() and (proba <= 1.0).all()

    def test_determinism_same_random_state(self):
        p1 = build_text_pipeline(random_state=3)
        p2 = build_text_pipeline(random_state=3)
        p1.fit(TEXT_X_TRAIN, TEXT_Y_TRAIN)
        p2.fit(TEXT_X_TRAIN, TEXT_Y_TRAIN)
        np.testing.assert_array_equal(
            p1.predict(TEXT_X_TRAIN),
            p2.predict(TEXT_X_TRAIN),
        )

    def test_unseen_text_predicted_without_error(self):
        preds = self.pipeline.predict(["Hello, this is a test message."])
        assert len(preds) == 1
        assert preds[0] in (0, 1)


# ===========================================================================
# Text pipeline — invalid inputs at predict time
# ===========================================================================


class TestTextPipelineInvalidPredict:
    def setup_method(self):
        self.pipeline = build_text_pipeline()
        self.pipeline.fit(TEXT_X_TRAIN, TEXT_Y_TRAIN)

    def test_plain_string_raises_typeerror(self):
        with pytest.raises(TypeError):
            self.pipeline.predict("a plain string")

    def test_empty_list_raises_valueerror(self):
        with pytest.raises(ValueError, match="empty"):
            self.pipeline.predict([])

    def test_list_of_ints_raises_typeerror(self):
        with pytest.raises(TypeError):
            self.pipeline.predict([1, 2, 3])


# ===========================================================================
# Text pipeline — invalid inputs at fit time
# ===========================================================================


class TestTextPipelineInvalidFit:
    def test_wrong_label_length_raises_valueerror(self):
        p = build_text_pipeline()
        with pytest.raises(ValueError, match="does not match"):
            p.fit(TEXT_X_TRAIN, [0, 1])

    def test_out_of_range_label_raises_valueerror(self):
        p = build_text_pipeline()
        bad_y = [0, 0, 0, 1, 1, 3]
        with pytest.raises(ValueError, match="0.*1"):
            p.fit(TEXT_X_TRAIN, bad_y)

    def test_non_list_raises_typeerror(self):
        p = build_text_pipeline()
        with pytest.raises(TypeError):
            p.fit("not a list at all", TEXT_Y_TRAIN)


# ===========================================================================
# Pipeline independence
# ===========================================================================


class TestPipelineIndependence:
    def test_url_and_text_pipelines_are_separate_objects(self):
        url_p = build_url_pipeline()
        txt_p = build_text_pipeline()
        assert url_p is not txt_p
        assert url_p.pipeline is not txt_p.pipeline

    def test_fitting_url_does_not_affect_text(self):
        url_p = build_url_pipeline()
        txt_p = build_text_pipeline()
        url_p.fit(URL_X_TRAIN, URL_Y_TRAIN)
        # Text pipeline must still be unfitted.
        from sklearn.exceptions import NotFittedError
        with pytest.raises(NotFittedError):
            txt_p.predict(TEXT_X_TRAIN)
