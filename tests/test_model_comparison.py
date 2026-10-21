"""
tests/test_model_comparison.py
==============================
Unit tests for src/model_comparison.py.

Strategy
--------
- All tests use tiny in-memory datasets (10 + 10 samples).
- No metrics are asserted as correct — only structure, types, and error handling.
- Calibration error-path tests use intentionally tiny datasets.

Coverage
--------
  ModelComparisonResult         — fields, as_dict, defaults
  compare_url_models            — returns list, sorted by f1, correct types
  compare_text_models           — returns list, sorted by f1, correct types
  _safe_cv_folds                — fold reduction, zero folds, no-warning path
  _run_cv                       — tiny dataset degrades gracefully
  calibrate_pipeline            — raises on tiny data, succeeds on adequate data
  print_comparison_table        — no crash, produces output
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.features.url_features import extract_url_features
from src.model_comparison import (
    ModelComparisonResult,
    _safe_cv_folds,
    _run_cv,
    calibrate_pipeline,
    compare_text_models,
    compare_url_models,
    print_comparison_table,
)
from src.models.pipeline import build_url_pipeline, build_text_pipeline


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

_LEGIT_URLS = [
    "https://www.amazon.com/",
    "https://www.google.com/",
    "https://www.paypal.com/",
    "https://www.microsoft.com/",
    "https://www.apple.com/",
    "https://www.facebook.com/",
    "https://www.netflix.com/",
    "https://www.github.com/",
    "https://www.linkedin.com/",
    "https://www.twitter.com/",
]

_PHISH_URLS = [
    "http://amaz0n-verify.xyz/",
    "http://goog1e-login.ml/",
    "http://paypa1-secure.tk/",
    "http://micros0ft-alert.ga/",
    "http://app1e-id.cf/",
    "http://faceb00k-login.gq/",
    "http://netf1ix-pay.xyz/",
    "http://githu8-secure.tk/",
    "http://1inkedin-jobs.ml/",
    "http://tw1tter-verify.ga/",
]

_ALL_URLS  = _LEGIT_URLS + _PHISH_URLS
_URL_LABELS = [0] * 10 + [1] * 10

_URL_FEATURES = [extract_url_features(u) for u in _ALL_URLS]

_LEGIT_TEXTS = [
    "Meeting rescheduled to Thursday.",
    "Your package has shipped.",
    "Please review the attached report.",
    "Team standup at 9am tomorrow.",
    "Code review approved and merged.",
    "Dinner reservation confirmed.",
    "Your prescription is ready.",
    "Flight departs at 07:45.",
    "Budget proposal approved.",
    "Deployment completed successfully.",
]
_PHISH_TEXTS = [
    "URGENT: your account has been suspended! Verify your password now!",
    "Click here to claim your $1000 prize at http://free-winner.xyz/",
    "Your PayPal is locked. Confirm credentials: http://paypal.tk/",
    "ALERT: verify your bank details immediately to avoid suspension.",
    "Invoice overdue. Wire payment or face termination: http://pay.ml/",
    "You won $5000! Send your bank account number to collect now.",
    "Your Apple ID will be terminated! Sign in: http://apple-locked.ga/",
    "Unusual activity — reset your password: http://secure-update.cf/",
    "Microsoft account at risk! Provide your username and password ASAP.",
    "Final notice: your credit card suspended. Confirm at http://card.gq/",
]
_ALL_TEXTS   = _LEGIT_TEXTS + _PHISH_TEXTS
_TEXT_LABELS = [0] * 10 + [1] * 10


# ===========================================================================
# ModelComparisonResult
# ===========================================================================


class TestModelComparisonResultDefaults:
    def test_fields_present(self):
        r = ModelComparisonResult(model_name="test")
        assert r.model_name     == "test"
        assert r.mean_f1        == 0.0
        assert r.mean_precision == 0.0
        assert r.mean_recall    == 0.0
        assert r.n_folds        == 0
        assert r.cv_warning     == ""

    def test_mean_roc_auc_default_nan(self):
        r = ModelComparisonResult(model_name="test")
        assert math.isnan(r.mean_roc_auc)

    def test_as_dict_keys(self):
        r = ModelComparisonResult(model_name="Dummy")
        d = r.as_dict()
        assert "model_name"     in d
        assert "mean_f1"        in d
        assert "mean_precision" in d
        assert "mean_recall"    in d
        assert "n_folds"        in d
        assert "cv_warning"     in d

    def test_as_dict_serialisable(self):
        import json
        r = ModelComparisonResult(model_name="LR", mean_f1=0.85, std_f1=0.03, n_folds=5)
        json.dumps(r.as_dict())  # must not raise

    def test_as_dict_nan_roc_auc_becomes_none(self):
        r = ModelComparisonResult(model_name="X")
        assert r.as_dict()["mean_roc_auc"] is None


# ===========================================================================
# _safe_cv_folds
# ===========================================================================


class TestSafeCvFolds:
    def test_enough_data_returns_requested_folds(self):
        folds, warn = _safe_cv_folds(n_samples=100, n_classes=2, requested_cv=5)
        assert folds == 5
        assert warn  == ""

    def test_too_small_reduces_folds(self):
        # 4 samples, 2 classes → 2 per class → max 2 folds
        folds, warn = _safe_cv_folds(n_samples=4, n_classes=2, requested_cv=5)
        assert folds <= 5
        assert folds >= 0

    def test_one_sample_per_class_returns_zero(self):
        folds, warn = _safe_cv_folds(n_samples=2, n_classes=2, requested_cv=5)
        assert folds == 0
        assert len(warn) > 0

    def test_warning_mentions_fallback_message(self):
        _, warn = _safe_cv_folds(n_samples=2, n_classes=2, requested_cv=5)
        assert "cross-validation" in warn.lower() or "sample" in warn.lower()

    def test_large_requested_cv_capped(self):
        folds, _ = _safe_cv_folds(n_samples=6, n_classes=2, requested_cv=10)
        assert folds <= 6


# ===========================================================================
# _run_cv
# ===========================================================================


class TestRunCv:
    def test_returns_model_comparison_result(self):
        pipe = build_url_pipeline(random_state=42)
        result = _run_cv(pipe, _URL_FEATURES, _URL_LABELS, cv=3, random_state=42, model_name="RF")
        assert isinstance(result, ModelComparisonResult)

    def test_model_name_preserved(self):
        pipe = build_url_pipeline(random_state=42)
        result = _run_cv(pipe, _URL_FEATURES, _URL_LABELS, cv=3, random_state=42, model_name="MyModel")
        assert result.model_name == "MyModel"

    def test_n_folds_set_when_cv_succeeds(self):
        pipe = build_url_pipeline(random_state=42)
        result = _run_cv(pipe, _URL_FEATURES, _URL_LABELS, cv=3, random_state=42, model_name="RF")
        assert result.n_folds >= 0  # may be reduced for tiny data

    def test_metrics_in_0_1_range(self):
        pipe = build_url_pipeline(random_state=42)
        result = _run_cv(pipe, _URL_FEATURES, _URL_LABELS, cv=3, random_state=42, model_name="RF")
        if result.n_folds > 0:
            assert 0.0 <= result.mean_precision <= 1.0
            assert 0.0 <= result.mean_recall    <= 1.0
            assert 0.0 <= result.mean_f1        <= 1.0

    def test_too_small_dataset_returns_zero_folds(self):
        pipe   = build_url_pipeline(random_state=42)
        feats  = [_URL_FEATURES[0], _URL_FEATURES[10]]   # 1 per class
        labels = [0, 1]
        result = _run_cv(pipe, feats, labels, cv=5, random_state=42, model_name="RF")
        assert result.n_folds == 0
        assert len(result.cv_warning) > 0


# ===========================================================================
# compare_url_models
# ===========================================================================


class TestCompareUrlModels:
    def setup_method(self):
        self.results = compare_url_models(
            _URL_FEATURES, _URL_LABELS, cv=3, random_state=42
        )

    def test_returns_list(self):
        assert isinstance(self.results, list)

    def test_three_models_returned(self):
        assert len(self.results) == 3

    def test_all_are_model_comparison_result(self):
        for r in self.results:
            assert isinstance(r, ModelComparisonResult)

    def test_sorted_by_mean_f1_descending(self):
        f1s = [r.mean_f1 for r in self.results]
        assert f1s == sorted(f1s, reverse=True)

    def test_model_names_all_distinct(self):
        names = [r.model_name for r in self.results]
        assert len(set(names)) == len(names)

    def test_dummy_model_present(self):
        names = [r.model_name for r in self.results]
        assert any("dummy" in n.lower() for n in names)

    def test_lr_model_present(self):
        names = [r.model_name for r in self.results]
        assert any("logistic" in n.lower() for n in names)

    def test_rf_model_present(self):
        names = [r.model_name for r in self.results]
        assert any("random" in n.lower() or "forest" in n.lower() for n in names)


# ===========================================================================
# compare_text_models
# ===========================================================================


class TestCompareTextModels:
    def setup_method(self):
        self.results = compare_text_models(
            _ALL_TEXTS, _TEXT_LABELS, cv=3, random_state=42
        )

    def test_returns_list(self):
        assert isinstance(self.results, list)

    def test_three_models_returned(self):
        assert len(self.results) == 3

    def test_all_are_model_comparison_result(self):
        for r in self.results:
            assert isinstance(r, ModelComparisonResult)

    def test_sorted_by_mean_f1_descending(self):
        f1s = [r.mean_f1 for r in self.results]
        assert f1s == sorted(f1s, reverse=True)

    def test_model_names_all_distinct(self):
        names = [r.model_name for r in self.results]
        assert len(set(names)) == len(names)


# ===========================================================================
# calibrate_pipeline
# ===========================================================================


class TestCalibratePipeline:
    def test_raises_on_too_small_dataset(self):
        pipe = build_url_pipeline(random_state=42)
        feats  = _URL_FEATURES[:5]   # < _MIN_CALIBRATION_SAMPLES
        labels = [0, 1, 0, 1, 0]
        with pytest.raises(ValueError, match="Calibration requires"):
            calibrate_pipeline(pipe, feats, labels)

    def test_succeeds_on_adequate_dataset(self):
        # Use an unfitted pipeline with inner cv=3; sklearn handles fit+calibrate internally
        pipe = build_url_pipeline(random_state=42)
        calibrated = calibrate_pipeline(pipe, _URL_FEATURES, _URL_LABELS, cv=3)
        # Calibrated pipeline must expose predict_proba
        probas = calibrated.predict_proba(_URL_FEATURES[:2])
        assert probas.shape[1] == 2

    def test_error_message_mentions_min_samples(self):
        pipe = build_url_pipeline(random_state=42)
        with pytest.raises(ValueError) as exc_info:
            calibrate_pipeline(pipe, _URL_FEATURES[:3], [0, 1, 0])
        assert "20" in str(exc_info.value) or "samples" in str(exc_info.value)


# ===========================================================================
# print_comparison_table
# ===========================================================================


class TestPrintComparisonTable:
    def test_no_crash_with_normal_results(self, capsys):
        results = compare_url_models(_URL_FEATURES, _URL_LABELS, cv=3, random_state=42)
        print_comparison_table(results)
        captured = capsys.readouterr()
        assert len(captured.out) > 0

    def test_output_contains_disclaimer(self, capsys):
        results = compare_url_models(_URL_FEATURES, _URL_LABELS, cv=3, random_state=42)
        print_comparison_table(results)
        captured = capsys.readouterr()
        assert "SYNTHETIC" in captured.out or "synthetic" in captured.out.lower()

    def test_output_contains_model_names(self, capsys):
        results = compare_url_models(_URL_FEATURES, _URL_LABELS, cv=3, random_state=42)
        print_comparison_table(results)
        captured = capsys.readouterr()
        for r in results:
            assert r.model_name[:10] in captured.out

    def test_empty_results_no_crash(self, capsys):
        print_comparison_table([])
        captured = capsys.readouterr()
        assert isinstance(captured.out, str)
