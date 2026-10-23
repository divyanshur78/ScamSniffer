from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# ruff: noqa: E402

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from scripts.train import (
    _split_stratified,
    _validate_columns,
    _validate_labels_column,
    _validate_min_class_counts,
    _validate_no_nulls,
    load_message_data,
    load_url_data,
    main,
    save_model,
    save_report,
    train_text_model,
    train_url_model,
)
from src.evaluation import EvaluationResult, evaluate


class TestEvaluateInputGuards:
    def test_empty_y_true_raises(self):
        with pytest.raises(ValueError, match="empty"):
            evaluate([], [])

    def test_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="same length"):
            evaluate([0, 1], [0])

    def test_out_of_range_y_true_raises(self):
        with pytest.raises(ValueError, match="outside"):
            evaluate([0, 2], [0, 1])

    def test_out_of_range_y_pred_raises(self):
        with pytest.raises(ValueError, match="outside"):
            evaluate([0, 1], [0, 3])


class TestEvaluateReturnStructure:
    def setup_method(self):
        self.result = evaluate(
            y_true=[0, 0, 1, 1],
            y_pred=[0, 0, 1, 1],
        )

    def test_returns_evaluation_result(self):
        assert isinstance(self.result, EvaluationResult)

    def test_precision_in_0_1(self):
        assert 0.0 <= self.result.precision <= 1.0

    def test_recall_in_0_1(self):
        assert 0.0 <= self.result.recall <= 1.0

    def test_f1_in_0_1(self):
        assert 0.0 <= self.result.f1 <= 1.0

    def test_confusion_matrix_is_2x2(self):
        cm = self.result.confusion_matrix
        assert len(cm) == 2
        assert all(len(row) == 2 for row in cm)

    def test_class_counts_keys(self):
        assert "legitimate (0)" in self.result.class_counts
        assert "phishing (1)" in self.result.class_counts

    def test_class_counts_sum_to_total(self):
        total = sum(self.result.class_counts.values())
        assert total == 4

    def test_classification_report_is_string(self):
        assert isinstance(self.result.classification_report, str)

    def test_roc_auc_none_without_proba(self):
        assert self.result.roc_auc is None


class TestEvaluateWithProba:
    def test_roc_auc_set_when_proba_supplied(self):
        result = evaluate(
            y_true=[0, 0, 1, 1],
            y_pred=[0, 0, 1, 1],
            y_proba=[0.1, 0.2, 0.8, 0.9],
        )
        assert result.roc_auc is not None
        assert 0.0 <= result.roc_auc <= 1.0

    def test_roc_auc_none_single_class(self):
        result = evaluate(
            y_true=[1, 1, 1],
            y_pred=[1, 1, 1],
            y_proba=[0.9, 0.85, 0.95],
        )
        assert result.roc_auc is None


class TestEvaluateAsDictSerializable:
    def test_as_dict_is_json_serializable(self):
        result = evaluate([0, 1, 0, 1], [0, 1, 1, 0])
        d = result.as_dict()
        serialized = json.dumps(d)
        assert isinstance(serialized, str)

    def test_as_dict_has_required_keys(self):
        result = evaluate([0, 1], [0, 1])
        keys = result.as_dict().keys()
        for expected in ("precision", "recall", "f1", "confusion_matrix",
                         "class_counts", "classification_report"):
            assert expected in keys


class TestEvaluatePerfectPrediction:
    def test_perfect_f1(self):
        result = evaluate([0, 0, 1, 1], [0, 0, 1, 1])
        assert result.f1 == pytest.approx(1.0)

    def test_perfect_precision(self):
        result = evaluate([0, 0, 1, 1], [0, 0, 1, 1])
        assert result.precision == pytest.approx(1.0)

    def test_perfect_recall(self):
        result = evaluate([0, 0, 1, 1], [0, 0, 1, 1])
        assert result.recall == pytest.approx(1.0)


class TestEvaluateAllWrongPrediction:
    def test_zero_precision_all_wrong(self):
        result = evaluate([1, 1, 1], [0, 0, 0])
        assert result.precision == pytest.approx(0.0)

    def test_zero_recall_all_wrong(self):
        result = evaluate([1, 1, 1], [0, 0, 0])
        assert result.recall == pytest.approx(0.0)


class TestValidateColumns:
    def test_passes_with_required_columns(self):
        df = pd.DataFrame({"url": ["x"], "label": [0]})
        _validate_columns(df, {"url", "label"}, "test.csv")

    def test_raises_on_missing_column(self):
        df = pd.DataFrame({"url": ["x"]})
        with pytest.raises(ValueError, match="missing required columns"):
            _validate_columns(df, {"url", "label"}, "test.csv")


class TestValidateLabelsColumn:
    def test_valid_binary_labels(self):
        df = pd.DataFrame({"label": [0, 1, 0, 1]})
        _validate_labels_column(df, "test.csv")

    def test_raises_on_bad_label(self):
        df = pd.DataFrame({"label": [0, 1, 2]})
        with pytest.raises(ValueError, match="outside"):
            _validate_labels_column(df, "test.csv")


class TestValidateNoNulls:
    def test_no_nulls_passes(self):
        df = pd.DataFrame({"url": ["http://x.com"], "label": [0]})
        _validate_no_nulls(df, "test.csv")

    def test_raises_on_null_value(self):
        df = pd.DataFrame({"url": [None], "label": [0]})
        with pytest.raises(ValueError, match="null values"):
            _validate_no_nulls(df, "test.csv")


class TestValidateMinClassCounts:
    def test_sufficient_counts_passes(self):
        s = pd.Series([0, 0, 1, 1])
        _validate_min_class_counts(s, "test.csv", minimum=2)

    def test_raises_when_class_missing(self):
        s = pd.Series([0, 0, 0])
        with pytest.raises(ValueError, match="class 1"):
            _validate_min_class_counts(s, "test.csv", minimum=1)

    def test_raises_when_count_below_minimum(self):
        s = pd.Series([0, 0, 0, 1])
        with pytest.raises(ValueError, match="class 1"):
            _validate_min_class_counts(s, "test.csv", minimum=2)

class TestLoadUrlData:
    def test_missing_file_raises_filenotfounderror(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_url_data(tmp_path)

    def test_missing_column_raises_valueerror(self, tmp_path):
        (tmp_path / "urls.csv").write_text("link,label\nhttp://x.com,0\n")
        with pytest.raises(ValueError, match="missing required columns"):
            load_url_data(tmp_path)

    def test_valid_csv_returns_four_tuple(self, tmp_path):
        csv = (
            "url,label\n"
            "https://example.com,0\n"
            "https://legit.org/page,0\n"
            "http://phish.xyz/login,1\n"
            "http://192.0.2.1/verify,1\n"
        )
        (tmp_path / "urls.csv").write_text(csv)
        result = load_url_data(tmp_path)
        assert len(result) == 4, (
            "load_url_data must return a 4-tuple "
            "(features, labels, urls, dedup_report)"
        )
        features, labels, raw_urls, dedup_report = result
        assert len(features) == len(labels)
        assert all(isinstance(f, dict) for f in features)
        assert set(labels).issubset({0, 1})
        assert isinstance(raw_urls, list)

    def test_valid_csv_returns_raw_urls(self, tmp_path):
        csv = (
            "url,label\n"
            "https://example.com,0\n"
            "https://legit.org,0\n"
            "http://phish.xyz/login,1\n"
            "http://evil.tk/verify,1\n"
        )
        (tmp_path / "urls.csv").write_text(csv)
        _, _, raw_urls, _ = load_url_data(tmp_path)
        assert all(isinstance(u, str) for u in raw_urls)

    def test_null_url_raises_valueerror(self, tmp_path):
        csv = "url,label\n,0\nhttps://example.com,1\n"
        (tmp_path / "urls.csv").write_text(csv)
        with pytest.raises(ValueError, match="null"):
            load_url_data(tmp_path)


class TestLoadMessageData:
    def test_missing_file_raises_filenotfounderror(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_message_data(tmp_path)

    def test_missing_text_column_raises(self, tmp_path):
        (tmp_path / "messages.csv").write_text("body,label\nhello,0\n")
        with pytest.raises(ValueError, match="missing required columns"):
            load_message_data(tmp_path)

    def test_valid_csv_returns_texts_and_labels(self, tmp_path):
        csv = (
            "text,type,label\n"
            '"Hello there.",email,0\n'
            '"Meeting at 3pm.",chat,0\n'
            '"URGENT: verify password!",email,1\n'
            '"Claim your prize now!",sms,1\n'
        )
        (tmp_path / "messages.csv").write_text(csv)
        texts, labels = load_message_data(tmp_path)
        assert len(texts) == 4
        assert labels == [0, 0, 1, 1]

class TestSplitStratified:
    def test_returns_two_non_overlapping_index_lists(self):
        labels = [0, 0, 0, 1, 1, 1]
        train, test = _split_stratified(6, labels, 0.33, 42)
        assert set(train).isdisjoint(set(test))
        assert sorted(train + test) == list(range(6))

    def test_all_classes_in_train_and_test(self):
        labels = [0, 0, 0, 0, 1, 1, 1, 1]
        train, test = _split_stratified(8, labels, 0.25, 42)
        train_labels = {labels[i] for i in train}
        test_labels  = {labels[i] for i in test}
        assert 0 in train_labels and 1 in train_labels
        assert 0 in test_labels  and 1 in test_labels

    def test_minimal_dataset_does_not_crash(self):
        # 2 samples per class — smallest valid input.
        labels = [0, 0, 1, 1]
        train, test = _split_stratified(4, labels, 0.5, 42)
        assert len(train) >= 1
        assert len(test) >= 1

_URL_FEATURES_LEGIT = [
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
_URL_FEATURES_PHISH = [
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
_URL_X = _URL_FEATURES_LEGIT + _URL_FEATURES_PHISH
_URL_Y = [0, 0, 0, 1, 1, 1]

_TEXT_X = [
    "Hi there, just checking in.",
    "Meeting rescheduled to Thursday.",
    "Your package has been delivered.",
    "URGENT: Your account is suspended! Verify password now!",
    "Click here to claim your prize: http://phishing.xyz/login",
    "Bank account locked. Confirm credentials now.",
]
_TEXT_Y = [0, 0, 0, 1, 1, 1]


class TestTrainUrlModel:
    def test_returns_pipeline_and_result(self):
        pipeline, result = train_url_model(_URL_X, _URL_Y, 0.33, 42)
        assert isinstance(result, EvaluationResult)

    def test_result_has_expected_metrics(self):
        _, result = train_url_model(_URL_X, _URL_Y, 0.33, 42)
        assert 0.0 <= result.precision <= 1.0
        assert 0.0 <= result.recall <= 1.0
        assert 0.0 <= result.f1 <= 1.0

    def test_confusion_matrix_shape(self):
        _, result = train_url_model(_URL_X, _URL_Y, 0.33, 42)
        assert len(result.confusion_matrix) == 2
        assert all(len(row) == 2 for row in result.confusion_matrix)

    def test_trained_pipeline_can_predict(self):
        pipeline, _ = train_url_model(_URL_X, _URL_Y, 0.33, 42)
        preds = pipeline.predict(_URL_X)
        assert len(preds) == len(_URL_X)
        assert set(preds.tolist()).issubset({0, 1})


class TestTrainTextModel:
    def test_returns_pipeline_and_result(self):
        pipeline, result = train_text_model(_TEXT_X, _TEXT_Y, 0.33, 42)
        assert isinstance(result, EvaluationResult)

    def test_result_has_expected_metrics(self):
        _, result = train_text_model(_TEXT_X, _TEXT_Y, 0.33, 42)
        assert 0.0 <= result.precision <= 1.0
        assert 0.0 <= result.recall <= 1.0
        assert 0.0 <= result.f1 <= 1.0

    def test_trained_pipeline_can_predict(self):
        pipeline, _ = train_text_model(_TEXT_X, _TEXT_Y, 0.33, 42)
        preds = pipeline.predict(_TEXT_X)
        assert len(preds) == len(_TEXT_X)
        assert set(preds.tolist()).issubset({0, 1})

class TestSaveHelpers:
    def test_save_model_creates_file(self, tmp_path):
        from src.models.pipeline import build_url_pipeline
        p = build_url_pipeline()
        p.fit(_URL_X, _URL_Y)
        out = tmp_path / "sub" / "model.joblib"
        save_model(p, out)
        assert out.exists()

    def test_save_report_creates_json(self, tmp_path):
        result = evaluate([0, 0, 1, 1], [0, 0, 1, 1])
        out = tmp_path / "report.json"
        save_report(result, out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert "precision" in data

    def test_save_report_is_valid_json(self, tmp_path):
        result = evaluate([0, 1, 0, 1], [0, 1, 1, 0])
        out = tmp_path / "metrics.json"
        save_report(result, out)
        json.loads(out.read_text())


class TestMainCli:
    def test_main_succeeds_on_sample_data(self, tmp_path):
        """End-to-end: main() runs on the real sample CSVs."""
        rc = main([
            "--data-dir",   str(Path("data/samples")),
            "--output-dir", str(tmp_path / "models"),
            "--report-dir", str(tmp_path / "reports"),
            "--random-state", "42",
            "--test-size",    "0.25",
        ])
        assert rc == 0

    def test_main_saves_model_files(self, tmp_path):
        rc = main([
            "--data-dir",   str(Path("data/samples")),
            "--output-dir", str(tmp_path / "models"),
            "--report-dir", str(tmp_path / "reports"),
        ])
        assert rc == 0
        assert (tmp_path / "models" / "url_model.joblib").exists()
        assert (tmp_path / "models" / "text_model.joblib").exists()

    def test_main_saves_report_files(self, tmp_path):
        rc = main([
            "--data-dir",   str(Path("data/samples")),
            "--output-dir", str(tmp_path / "models"),
            "--report-dir", str(tmp_path / "reports"),
        ])
        assert rc == 0
        assert (tmp_path / "reports" / "url_evaluation.json").exists()
        assert (tmp_path / "reports" / "text_evaluation.json").exists()

    def test_main_returns_1_on_missing_data_dir(self, tmp_path):
        rc = main([
            "--data-dir",   str(tmp_path / "nonexistent"),
            "--output-dir", str(tmp_path / "models"),
            "--report-dir", str(tmp_path / "reports"),
        ])
        assert rc == 1

    def test_report_json_has_expected_keys(self, tmp_path):
        main([
            "--data-dir",   str(Path("data/samples")),
            "--output-dir", str(tmp_path / "models"),
            "--report-dir", str(tmp_path / "reports"),
        ])
        data = json.loads(
            (tmp_path / "reports" / "url_evaluation.json").read_text()
        )
        for key in ("precision", "recall", "f1", "confusion_matrix"):
            assert key in data
