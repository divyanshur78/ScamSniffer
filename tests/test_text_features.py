
from __future__ import annotations

import pytest

from src.features.text_features import (
    _count_terms,
    _digit_ratio,
    _unique_word_ratio,
    _uppercase_ratio,
    extract_text_features,
)
from src.validator import MESSAGE_MAX_LENGTH

EXPECTED_KEYS = {
    "text_length",
    "word_count",
    "avg_word_length",
    "uppercase_ratio",
    "exclamation_count",
    "question_count",
    "url_count",
    "urgency_term_count",
    "credential_req_count",
    "money_term_count",
    "digit_ratio",
    "unique_word_ratio",
}

class TestUppercaseRatio:
    def test_all_lowercase_is_zero(self):
        assert _uppercase_ratio("hello world") == 0.0

    def test_all_uppercase_is_one(self):
        assert _uppercase_ratio("HELLO") == 1.0

    def test_half_uppercase(self):
        assert _uppercase_ratio("Ab") == pytest.approx(0.5)

    def test_no_alpha_chars_returns_zero(self):
        assert _uppercase_ratio("12345!?") == 0.0

    def test_digits_ignored(self):
        assert _uppercase_ratio("A1B") == 1.0


class TestDigitRatio:
    def test_no_digits_returns_zero(self):
        assert _digit_ratio("hello") == 0.0

    def test_all_digits_returns_one(self):
        assert _digit_ratio("123456") == 1.0

    def test_mixed(self):
        assert _digit_ratio("a1") == pytest.approx(0.5)

    def test_empty_returns_zero(self):
        assert _digit_ratio("") == 0.0


class TestUniqueWordRatio:
    def test_all_unique_returns_one(self):
        assert _unique_word_ratio(["hello", "world", "foo"]) == 1.0

    def test_all_same_returns_fraction(self):
        assert _unique_word_ratio(["win", "win", "win"]) == pytest.approx(1 / 3)

    def test_empty_list_returns_zero(self):
        assert _unique_word_ratio([]) == 0.0

    def test_case_insensitive(self):
        assert _unique_word_ratio(["WIN", "win"]) == pytest.approx(0.5)


class TestCountTerms:
    def test_no_match_returns_zero(self):
        assert _count_terms("hello world", frozenset({"danger"})) == 0

    def test_single_match(self):
        assert _count_terms("your account is suspended", frozenset({"suspended"})) == 1

    def test_multiple_distinct_terms_counted_once_each(self):
        text = "urgent: verify your password now"
        count = _count_terms(text, frozenset({"urgent", "verify", "password"}))
        assert count == 3

    def test_repeated_term_counted_once(self):
        count = _count_terms("urgent urgent", frozenset({"urgent"}))
        assert count == 1

    def test_multi_word_phrase(self):
        count = _count_terms("click here to claim your prize", frozenset({"click here"}))
        assert count == 1

    def test_word_boundary_respected(self):
        count = _count_terms("spinning wheel", frozenset({"pin"}))
        assert count == 0


class TestFeatureDictShape:
    def test_returns_dict(self):
        assert isinstance(extract_text_features("Hello, this is a normal message."), dict)

    def test_contains_all_expected_keys(self):
        result = extract_text_features("Hello world.")
        assert result.keys() == EXPECTED_KEYS

    def test_all_values_are_numeric(self):
        result = extract_text_features("Hello world.")
        for key, val in result.items():
            assert isinstance(val, (int, float)), f"{key} has non-numeric value {val!r}"

    def test_no_none_values(self):
        result = extract_text_features("Hello world.")
        assert all(v is not None for v in result.values())

class TestNormalMessage:
    MSG = "Hi there, just checking in to see how you are doing today."

    def setup_method(self):
        self.f = extract_text_features(self.MSG)

    def test_text_length_matches(self):
        assert self.f["text_length"] == len(self.MSG)

    def test_word_count_correct(self):
        assert self.f["word_count"] == len(self.MSG.split())

    def test_no_exclamations(self):
        assert self.f["exclamation_count"] == 0

    def test_no_urls(self):
        assert self.f["url_count"] == 0

    def test_low_urgency(self):
        assert self.f["urgency_term_count"] == 0

    def test_low_credential_count(self):
        assert self.f["credential_req_count"] == 0

    def test_low_money_count(self):
        assert self.f["money_term_count"] == 0

    def test_uppercase_ratio_low(self):
        assert self.f["uppercase_ratio"] < 0.5

class TestSuspiciousPhishingMessage:
    MSG = (
        "URGENT: Your account has been suspended! "
        "Verify your password and bank account credentials immediately. "
        "Click here to claim your prize: http://phishing.xyz/login?verify=1"
    )

    def setup_method(self):
        self.f = extract_text_features(self.MSG)

    def test_high_urgency_count(self):
        assert self.f["urgency_term_count"] >= 2

    def test_high_credential_count(self):
        assert self.f["credential_req_count"] >= 2

    def test_high_money_count(self):
        assert self.f["money_term_count"] >= 2

    def test_has_exclamation(self):
        assert self.f["exclamation_count"] >= 1

    def test_has_url(self):
        assert self.f["url_count"] == 1

    def test_uppercase_ratio_elevated(self):
        assert self.f["uppercase_ratio"] > 0


class TestSmsMessage:
    def test_short_message_word_count(self):
        f = extract_text_features("Win now! Call 555-1234.", input_type="SMS")
        assert f["word_count"] == 4

    def test_exclamation_detected(self):
        f = extract_text_features("Win now!", input_type="SMS")
        assert f["exclamation_count"] == 1

    def test_digit_ratio_positive(self):
        f = extract_text_features("Call 555-1234 today.", input_type="SMS")
        assert f["digit_ratio"] > 0.0

class TestChatMessageWithUrls:
    def test_single_url_detected(self):
        f = extract_text_features(
            "Check this out: https://example.com/page", input_type="chat message"
        )
        assert f["url_count"] == 1

    def test_multiple_urls_detected(self):
        f = extract_text_features(
            "Visit http://a.com and https://b.com for details.",
            input_type="chat message",
        )
        assert f["url_count"] == 2

    def test_no_url_gives_zero(self):
        f = extract_text_features("No links here.", input_type="chat message")
        assert f["url_count"] == 0

    def test_url_not_fetched(self):
        # Purely structural: this test verifies the function completes without
        # making any network call (it would raise or hang if it tried).
        f = extract_text_features("See http://192.168.0.1/admin for details.")
        assert f["url_count"] == 1

class TestUppercaseRatioFeature:
    def test_all_caps_message(self):
        f = extract_text_features("CLICK HERE NOW")
        assert f["uppercase_ratio"] == pytest.approx(1.0)

    def test_all_lower_message(self):
        f = extract_text_features("click here now")
        assert f["uppercase_ratio"] == pytest.approx(0.0)

    def test_digits_only_message_is_zero(self):
        f = extract_text_features("12345678")
        assert f["uppercase_ratio"] == pytest.approx(0.0)


class TestPunctuation:
    def test_multiple_exclamations(self):
        f = extract_text_features("Win!!! Act now!!")
        assert f["exclamation_count"] == 5

    def test_question_marks(self):
        f = extract_text_features("Are you sure? Really?")
        assert f["question_count"] == 2

    def test_both_punctuation(self):
        f = extract_text_features("What?! Act now!")
        assert f["exclamation_count"] == 2
        assert f["question_count"] == 1

class TestUrgencyTerms:
    @pytest.mark.parametrize("term", ["urgent", "suspended", "alert", "warning", "critical"])
    def test_known_urgency_term_detected(self, term):
        f = extract_text_features(f"Your account is {term}.")
        assert f["urgency_term_count"] >= 1


class TestCredentialTerms:
    @pytest.mark.parametrize("term", ["password", "verify", "login", "confirm", "otp"])
    def test_known_credential_term_detected(self, term):
        f = extract_text_features(f"Please {term} your details.")
        assert f["credential_req_count"] >= 1


class TestMoneyTerms:
    @pytest.mark.parametrize("term", ["payment", "invoice", "prize", "winner", "refund"])
    def test_known_money_term_detected(self, term):
        f = extract_text_features(f"You have a pending {term}.")
        assert f["money_term_count"] >= 1

class TestUniqueWordRatioFeature:
    def test_repetitive_text_has_low_ratio(self):
        f = extract_text_features("win win win win win")
        assert f["unique_word_ratio"] == pytest.approx(1 / 5)

    def test_all_unique_words_has_ratio_one(self):
        f = extract_text_features("the quick brown fox")
        assert f["unique_word_ratio"] == pytest.approx(1.0)

class TestWhitespaceHandling:
    def test_leading_trailing_whitespace_stripped(self):
        f = extract_text_features("  hello world  ")
        assert f["text_length"] == len("hello world")

    def test_word_count_unaffected_by_leading_space(self):
        f = extract_text_features("  hello world  ")
        assert f["word_count"] == 2

class TestAvgWordLength:
    def test_single_word(self):
        f = extract_text_features("hello")
        assert f["avg_word_length"] == pytest.approx(5.0)

    def test_two_equal_length_words(self):
        # "hi" + "ho" → (2+2)/2 = 2.0
        f = extract_text_features("hi ho")
        assert f["avg_word_length"] == pytest.approx(2.0)

class TestInvalidInputs:
    def test_none_raises_valueerror(self):
        with pytest.raises(ValueError):
            extract_text_features(None)

    def test_integer_raises_valueerror(self):
        with pytest.raises(ValueError):
            extract_text_features(42)

    def test_list_raises_valueerror(self):
        with pytest.raises(ValueError):
            extract_text_features(["spam"])

    def test_empty_string_raises_valueerror(self):
        with pytest.raises(ValueError):
            extract_text_features("")

    def test_whitespace_only_raises_valueerror(self):
        with pytest.raises(ValueError):
            extract_text_features("   ")

    def test_oversized_message_raises_valueerror(self):
        with pytest.raises(ValueError):
            extract_text_features("a" * (MESSAGE_MAX_LENGTH + 1))

    def test_custom_input_type_in_error(self):
        with pytest.raises(ValueError, match="Sms"):
            extract_text_features("", input_type="SMS")
