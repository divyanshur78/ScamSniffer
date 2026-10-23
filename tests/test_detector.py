
from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.detector import (
    _DISCLAIMER,
    DetectionResult,
    Detector,
    InputType,
    ModelNotReadyError,
    _text_warnings,
    _url_warnings,
    detect_message,
    detect_url,
)

_MODEL_DIR = Path("models")


@pytest.fixture(scope="session")
def detector() -> Detector:
    return Detector(model_dir=_MODEL_DIR)

class TestInputType:
    def test_url_value(self):
        assert InputType.URL.value == "url"

    def test_email_value(self):
        assert InputType.EMAIL.value == "email"

    def test_sms_value(self):
        assert InputType.SMS.value == "sms"

    def test_chat_value(self):
        assert InputType.CHAT.value == "chat"

    def test_case_insensitive_routing(self, detector):
        r = detector.detect("https://example.com/page", "URL")
        assert r.input_type == "url"

    def test_unknown_type_raises_valueerror(self, detector):
        with pytest.raises(ValueError, match="Unknown input_type"):
            detector.detect("https://example.com", "fax")


class TestDetectorConstruction:
    def test_loads_from_real_model_dir(self):
        d = Detector(model_dir=_MODEL_DIR)
        assert d is not None

    def test_missing_url_model_raises(self, tmp_path):
        import shutil
        shutil.copy(_MODEL_DIR / "text_model.joblib", tmp_path / "text_model.joblib")
        with pytest.raises(ModelNotReadyError, match="url_model.joblib"):
            Detector(model_dir=tmp_path)

    def test_missing_text_model_raises(self, tmp_path):
        import shutil
        shutil.copy(_MODEL_DIR / "url_model.joblib", tmp_path / "url_model.joblib")
        with pytest.raises(ModelNotReadyError, match="text_model.joblib"):
            Detector(model_dir=tmp_path)

    def test_empty_model_dir_raises(self, tmp_path):
        with pytest.raises(ModelNotReadyError):
            Detector(model_dir=tmp_path)

    def test_error_message_contains_train_hint(self, tmp_path):
        with pytest.raises(ModelNotReadyError, match="scripts/train.py"):
            Detector(model_dir=tmp_path)

class TestDetectionResultStructure:

    _VALID_VERDICTS = {
        "Potentially phishing",
        "No strong signals detected",
        "Unable to assess",
    }

    def _check(self, result: DetectionResult, expected_type: str) -> None:
        assert isinstance(result, DetectionResult)
        assert result.input_type == expected_type
        assert result.label in (0, 1)
        assert result.label_text in ("legitimate", "phishing")
        assert result.label_text == ("legitimate" if result.label == 0 else "phishing")
        assert hasattr(result, "verdict"), "DetectionResult must have a 'verdict' field"
        assert result.verdict in self._VALID_VERDICTS, (
            f"verdict {result.verdict!r} is not one of {self._VALID_VERDICTS}"
        )
        assert 0.0 <= result.confidence <= 1.0
        assert 0.0 <= result.phishing_prob <= 1.0
        assert isinstance(result.warnings, list)
        assert isinstance(result.disclaimer, str)
        assert len(result.disclaimer) > 0

    def test_url_result_structure(self, detector):
        r = detector.detect("https://example.com/page", InputType.URL)
        self._check(r, "url")

    def test_email_result_structure(self, detector):
        r = detector.detect("Hi, how are you doing today?", InputType.EMAIL)
        self._check(r, "email")

    def test_sms_result_structure(self, detector):
        r = detector.detect("Your package is on its way.", InputType.SMS)
        self._check(r, "sms")

    def test_chat_result_structure(self, detector):
        r = detector.detect("The meeting is at 3pm.", InputType.CHAT)
        self._check(r, "chat")

    def test_disclaimer_is_always_present(self, detector):
        r = detector.detect("https://example.com", InputType.URL)
        assert _DISCLAIMER in r.disclaimer

    def test_confidence_matches_predicted_class(self, detector):
        r = detector.detect("https://example.com/page", InputType.URL)
        if r.label == 0:
            assert r.confidence == pytest.approx(1.0 - r.phishing_prob, abs=1e-4)
        else:
            assert r.confidence == pytest.approx(r.phishing_prob, abs=1e-4)


class TestAsDictSerialization:
    def test_as_dict_is_json_serializable(self, detector):
        r = detector.detect("https://example.com", InputType.URL)
        serialized = json.dumps(r.as_dict())  # must not raise
        assert isinstance(serialized, str)

    def test_as_dict_has_required_keys(self, detector):
        r = detector.detect("Hello there.", InputType.EMAIL)
        d = r.as_dict()
        for key in ("input_type", "label", "label_text", "verdict",
                    "confidence", "phishing_prob", "warnings", "disclaimer"):
            assert key in d

    def test_as_dict_verdict_is_valid(self, detector):
        r = detector.detect("https://example.com", InputType.URL)
        d = r.as_dict()
        assert d["verdict"] in {
            "Potentially phishing",
            "No strong signals detected",
            "Unable to assess",
        }

    def test_as_dict_confidence_rounded(self, detector):
        r = detector.detect("https://example.com", InputType.URL)
        d = r.as_dict()
        assert len(str(d["confidence"]).rstrip("0").split(".")[-1]) <= 4

class TestUrlDetection:
    def test_benign_url_returns_url_type(self, detector):
        r = detector.detect("https://www.python.org/downloads/", InputType.URL)
        assert r.input_type == "url"

    def test_suspicious_url_returns_url_type(self, detector):
        r = detector.detect("http://paypal-login.xyz/verify?account=1", InputType.URL)
        assert r.input_type == "url"

    def test_url_with_ip_host_produces_warning(self, detector):
        r = detector.detect("http://192.168.0.1/login", InputType.URL)
        assert any("IP address" in w for w in r.warnings)

    def test_http_url_produces_https_warning(self, detector):
        r = detector.detect("http://example.com/page", InputType.URL)
        assert any("HTTP" in w for w in r.warnings)

    def test_suspicious_tld_produces_warning(self, detector):
        r = detector.detect("http://example.xyz/login", InputType.URL)
        assert any("top-level domain" in w.lower() for w in r.warnings)

    def test_url_never_fetched(self, detector):
        r = detector.detect("http://192.0.2.0/should-not-be-fetched", InputType.URL)
        assert r.input_type == "url"

    def test_invalid_url_none_raises_valueerror(self, detector):
        with pytest.raises(ValueError):
            detector.detect(None, InputType.URL)

    def test_invalid_url_no_scheme_raises_valueerror(self, detector):
        with pytest.raises(ValueError):
            detector.detect("example.com/page", InputType.URL)

    def test_invalid_url_empty_raises_valueerror(self, detector):
        with pytest.raises(ValueError):
            detector.detect("", InputType.URL)

    def test_invalid_url_wrong_type_raises_valueerror(self, detector):
        with pytest.raises(ValueError):
            detector.detect(12345, InputType.URL)

class TestEmailDetection:
    def test_benign_email_returns_email_type(self, detector):
        r = detector.detect(
            "Hi Alice, just checking in. See you Thursday.", InputType.EMAIL
        )
        assert r.input_type == "email"

    def test_suspicious_email_type(self, detector):
        r = detector.detect(
            "URGENT: Your account is suspended! Verify your password now!",
            InputType.EMAIL,
        )
        assert r.input_type == "email"

    def test_suspicious_email_has_urgency_warning(self, detector):
        r = detector.detect(
            "URGENT: Your account is suspended! Verify your password now!",
            InputType.EMAIL,
        )
        assert any("urgency" in w.lower() for w in r.warnings)

    def test_credential_request_produces_warning(self, detector):
        r = detector.detect(
            "Please confirm your password and login credentials immediately.",
            InputType.EMAIL,
        )
        assert any("credential" in w.lower() for w in r.warnings)

    def test_money_term_produces_warning(self, detector):
        r = detector.detect(
            "You have won a prize! Claim your payment now.", InputType.EMAIL
        )
        assert any(
            "financial" in w.lower()
            or "prize" in w.lower()
            or "money" in w.lower()
            for w in r.warnings
        )
    def test_invalid_email_none_raises(self, detector):
        with pytest.raises(ValueError):
            detector.detect(None, InputType.EMAIL)

    def test_invalid_email_empty_raises(self, detector):
        with pytest.raises(ValueError):
            detector.detect("", InputType.EMAIL)


class TestSmsDetection:
    def test_benign_sms_type(self, detector):
        r = detector.detect("Your delivery arrives Tuesday.", InputType.SMS)
        assert r.input_type == "sms"

    def test_suspicious_sms_has_warnings(self, detector):
        r = detector.detect(
            "WIN $1000! Click here: http://prize.xyz/claim. Limited time!",
            InputType.SMS,
        )
        assert len(r.warnings) >= 1

    def test_sms_with_embedded_url_produces_warning(self, detector):
        r = detector.detect(
            "Check this link: https://example.com/page", InputType.SMS
        )
        assert any("URL" in w for w in r.warnings)

class TestChatDetection:
    def test_benign_chat_type(self, detector):
        r = detector.detect("The PR has been merged.", InputType.CHAT)
        assert r.input_type == "chat"

    def test_suspicious_chat_has_warnings(self, detector):
        r = detector.detect(
            "Send me your password and otp right now. Do not ignore this alert.",
            InputType.CHAT,
        )
        assert any("credential" in w.lower() or "urgency" in w.lower() for w in r.warnings)

class TestUrlWarnings:
    def test_ip_host_warning(self):
        f = {"is_ip_host": 1, "is_https": 1, "has_suspicious_tld": 0,
             "has_non_std_port": 0, "subdomain_depth": 0, "domain_entropy": 2.0,
             "suspicious_kw_count": 0, "url_length": 30}
        ws = _url_warnings(f, "http://192.168.0.1/page")
        assert any("IP address" in w for w in ws)

    def test_http_warning(self):
        f = {"is_ip_host": 0, "is_https": 0, "has_suspicious_tld": 0,
             "has_non_std_port": 0, "subdomain_depth": 0, "domain_entropy": 2.0,
             "suspicious_kw_count": 0, "url_length": 30}
        ws = _url_warnings(f, "http://example.com/page")
        assert any("HTTP" in w for w in ws)

    def test_no_warnings_for_clean_url(self):
        f = {"is_ip_host": 0, "is_https": 1, "has_suspicious_tld": 0,
             "has_non_std_port": 0, "subdomain_depth": 1, "domain_entropy": 2.5,
             "suspicious_kw_count": 0, "url_length": 35}
        ws = _url_warnings(f, "https://www.example.com/page")
        assert ws == []

    def test_suspicious_tld_warning(self):
        f = {"is_ip_host": 0, "is_https": 0, "has_suspicious_tld": 1,
             "has_non_std_port": 0, "subdomain_depth": 0, "domain_entropy": 2.0,
             "suspicious_kw_count": 0, "url_length": 30}
        ws = _url_warnings(f, "http://example.xyz/page")
        assert any("top-level domain" in w.lower() for w in ws)

    def test_high_keyword_count_warning(self):
        f = {"is_ip_host": 0, "is_https": 0, "has_suspicious_tld": 0,
             "has_non_std_port": 0, "subdomain_depth": 0, "domain_entropy": 2.0,
             "suspicious_kw_count": 3, "url_length": 50}
        ws = _url_warnings(f, "http://example.com/login-verify-account")
        assert any("keyword" in w.lower() for w in ws)


class TestTextWarnings:
    def test_urgency_warning(self):
        f = {"urgency_term_count": 2, "credential_req_count": 0,
             "money_term_count": 0, "url_count": 0,
             "uppercase_ratio": 0.1, "exclamation_count": 0}
        ws = _text_warnings(f, "Your account is suspended. Alert!")
        assert any("urgency" in w.lower() for w in ws)

    def test_credential_warning(self):
        f = {"urgency_term_count": 0, "credential_req_count": 1,
             "money_term_count": 0, "url_count": 0,
             "uppercase_ratio": 0.0, "exclamation_count": 0}
        ws = _text_warnings(f, "Please verify your password now.")
        assert any("credential" in w.lower() for w in ws)

    def test_embedded_url_warning(self):
        f = {"urgency_term_count": 0, "credential_req_count": 0,
             "money_term_count": 0, "url_count": 2,
             "uppercase_ratio": 0.0, "exclamation_count": 0}
        ws = _text_warnings(f, "Visit http://a.com and http://b.com.")
        assert any("URL" in w for w in ws)

    def test_no_warnings_for_clean_text(self):
        f = {"urgency_term_count": 0, "credential_req_count": 0,
             "money_term_count": 0, "url_count": 0,
             "uppercase_ratio": 0.1, "exclamation_count": 0}
        ws = _text_warnings(f, "Hi there, just checking in.")
        assert ws == []

    def test_high_uppercase_warning(self):
        f = {"urgency_term_count": 0, "credential_req_count": 0,
             "money_term_count": 0, "url_count": 0,
             "uppercase_ratio": 0.9, "exclamation_count": 0}
        ws = _text_warnings(f, "CLICK HERE NOW")
        assert any("uppercase" in w.lower() for w in ws)

class TestConvenienceFunctions:
    def test_detect_url_returns_result(self):
        r = detect_url("https://example.com/page", model_dir=_MODEL_DIR)
        assert isinstance(r, DetectionResult)
        assert r.input_type == "url"

    def test_detect_message_email_returns_result(self):
        r = detect_message(
            "Hello, how are you?", input_type="email", model_dir=_MODEL_DIR
        )
        assert isinstance(r, DetectionResult)
        assert r.input_type == "email"

    def test_detect_message_sms_returns_result(self):
        r = detect_message(
            "Your package is ready.", input_type="sms", model_dir=_MODEL_DIR
        )
        assert r.input_type == "sms"

    def test_detect_message_chat_returns_result(self):
        r = detect_message(
            "PR approved!", input_type="chat", model_dir=_MODEL_DIR
        )
        assert r.input_type == "chat"

    def test_detect_url_missing_model_raises(self, tmp_path):
        with pytest.raises(ModelNotReadyError):
            detect_url("https://example.com", model_dir=tmp_path)

    def test_detect_message_missing_model_raises(self, tmp_path):
        with pytest.raises(ModelNotReadyError):
            detect_message("Hello.", model_dir=tmp_path)
