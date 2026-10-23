
from __future__ import annotations

import pytest

from src.validator import (
    MESSAGE_MAX_LENGTH,
    URL_MAX_LENGTH,
    validate_message,
    validate_url,
)


class TestValidateUrlValid:
    def test_http_url_returned_unchanged(self):
        url = "http://example.com/path?q=1"
        assert validate_url(url) == url

    def test_https_url_returned_unchanged(self):
        url = "https://example.com"
        assert validate_url(url) == url

    def test_leading_trailing_whitespace_stripped(self):
        assert validate_url("  https://example.com  ") == "https://example.com"

    def test_url_at_exact_max_length(self):
        base = "https://example.com/"
        padding = "a" * (URL_MAX_LENGTH - len(base))
        url = base + padding
        assert len(url) == URL_MAX_LENGTH
        result = validate_url(url)
        assert result == url

    def test_uppercase_scheme_accepted(self):
        assert validate_url("HTTPS://example.com") == "HTTPS://example.com"

    def test_mixed_case_scheme_accepted(self):
        assert validate_url("Http://example.com") == "Http://example.com"

class TestValidateUrlRejection:
    def test_none_raises_valueerror(self):
        with pytest.raises(ValueError, match="None"):
            validate_url(None)

    def test_integer_raises_valueerror(self):
        with pytest.raises(ValueError, match="int"):
            validate_url(42)

    def test_list_raises_valueerror(self):
        with pytest.raises(ValueError, match="list"):
            validate_url(["https://example.com"])

    def test_empty_string_raises_valueerror(self):
        with pytest.raises(ValueError, match="empty or whitespace"):
            validate_url("")

    def test_whitespace_only_raises_valueerror(self):
        with pytest.raises(ValueError, match="empty or whitespace"):
            validate_url("   ")

    def test_newline_only_raises_valueerror(self):
        with pytest.raises(ValueError, match="empty or whitespace"):
            validate_url("\n\t")

    def test_oversized_url_raises_valueerror(self):
        long_url = "https://example.com/" + "a" * URL_MAX_LENGTH
        with pytest.raises(ValueError, match=str(URL_MAX_LENGTH)):
            validate_url(long_url)

    def test_ftp_scheme_rejected(self):
        with pytest.raises(ValueError, match="http"):
            validate_url("ftp://example.com/file.txt")

    def test_no_scheme_rejected(self):
        with pytest.raises(ValueError, match="http"):
            validate_url("example.com")

    def test_javascript_scheme_rejected(self):
        with pytest.raises(ValueError, match="http"):
            validate_url("javascript:alert(1)")

    def test_data_uri_rejected(self):
        with pytest.raises(ValueError, match="http"):
            validate_url("data:text/html,<h1>hi</h1>")

    def test_float_raises_valueerror(self):
        with pytest.raises(ValueError, match="float"):
            validate_url(3.14)

    def test_bool_raises_valueerror(self):
        with pytest.raises(ValueError, match="bool"):
            validate_url(True)

class TestValidateMessageValid:
    def test_plain_text_returned(self):
        msg = "Click here to win a prize!"
        assert validate_message(msg) == msg

    def test_leading_trailing_whitespace_stripped(self):
        assert validate_message("  hello world  ") == "hello world"

    def test_multiline_message_accepted(self):
        msg = "Dear user,\nYour account has been compromised.\nClick here."
        assert validate_message(msg) == msg

    def test_message_at_exact_max_length(self):
        msg = "a" * MESSAGE_MAX_LENGTH
        assert validate_message(msg) == msg

    def test_custom_input_type_returned_correctly(self):
        msg = "Win a free iPhone now!"
        assert validate_message(msg, input_type="SMS") == msg

    def test_email_input_type_label(self):
        msg = "Verify your account immediately."
        assert validate_message(msg, input_type="email") == msg

    def test_chat_input_type_label(self):
        msg = "Send me your password."
        assert validate_message(msg, input_type="chat message") == msg

class TestValidateMessageRejection:
    def test_none_raises_valueerror(self):
        with pytest.raises(ValueError, match="None"):
            validate_message(None)

    def test_integer_raises_valueerror(self):
        with pytest.raises(ValueError, match="int"):
            validate_message(99)

    def test_list_raises_valueerror(self):
        with pytest.raises(ValueError, match="list"):
            validate_message(["spam"])

    def test_dict_raises_valueerror(self):
        with pytest.raises(ValueError, match="dict"):
            validate_message({"text": "spam"})

    def test_empty_string_raises_valueerror(self):
        with pytest.raises(ValueError, match="empty or whitespace"):
            validate_message("")

    def test_whitespace_only_raises_valueerror(self):
        with pytest.raises(ValueError, match="empty or whitespace"):
            validate_message("     ")

    def test_tab_only_raises_valueerror(self):
        with pytest.raises(ValueError, match="empty or whitespace"):
            validate_message("\t\t\t")

    def test_oversized_message_raises_valueerror(self):
        long_msg = "a" * (MESSAGE_MAX_LENGTH + 1)
        with pytest.raises(ValueError, match=str(MESSAGE_MAX_LENGTH)):
            validate_message(long_msg)

    def test_custom_label_appears_in_error(self):
        with pytest.raises(ValueError, match="Sms"):
            validate_message("", input_type="SMS")

    def test_float_raises_valueerror(self):
        with pytest.raises(ValueError, match="float"):
            validate_message(1.5)
