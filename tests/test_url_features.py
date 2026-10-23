
from __future__ import annotations

import math

import pytest

from src.features.url_features import (
    _KNOWN_BRANDS,
    _brand_in_subdomain,
    _brand_min_edit_distance,
    _count_suspicious_keywords,
    _has_digit_substitution,
    _has_hyphen_brand,
    _has_redirect_param,
    _has_userinfo,
    _is_ip_address,
    _levenshtein,
    _longest_consecutive_digits,
    _path_has_brand,
    _shannon_entropy,
    _tld_label_count,
    extract_url_features,
)
from src.validator import URL_MAX_LENGTH

ORIGINAL_KEYS = {
    "url_length",
    "hostname_length",
    "path_length",
    "query_length",
    "dot_count",
    "subdomain_depth",
    "domain_entropy",
    "is_ip_host",
    "is_https",
    "suspicious_kw_count",
    "has_non_std_port",
    "has_suspicious_tld",
}

NEW_KEYS = {
    "hostname_label_count",
    "path_depth",
    "has_at_in_authority",
    "has_encoded_chars",
    "has_redirect_param",
    "is_idn",
    "brand_min_edit_dist",
    "brand_in_subdomain",
    "has_digit_sub_in_domain",
    "has_hyphen_brand",
    "consecutive_digits",
    "has_userinfo",
    "path_has_brand",
    "tld_label_count",
}

EXPECTED_KEYS = ORIGINAL_KEYS | NEW_KEYS

class TestShannonEntropy:
    def test_empty_string_returns_zero(self):
        assert _shannon_entropy("") == 0.0

    def test_single_char_returns_zero(self):
        assert _shannon_entropy("a") == 0.0

    def test_two_equal_chars_returns_one(self):
        assert math.isclose(_shannon_entropy("ab"), 1.0, rel_tol=1e-9)

    def test_uniform_four_chars(self):
        assert math.isclose(_shannon_entropy("abcd"), 2.0, rel_tol=1e-9)

    def test_repeated_char_is_lower_entropy(self):
        assert _shannon_entropy("aaab") < _shannon_entropy("abcd")


class TestIsIpAddress:
    def test_valid_ipv4(self):
        assert _is_ip_address("192.168.1.1") is True

    def test_all_zeros_ipv4(self):
        assert _is_ip_address("0.0.0.0") is True

    def test_broadcast_ipv4(self):
        assert _is_ip_address("255.255.255.255") is True

    def test_hostname_is_not_ip(self):
        assert _is_ip_address("example.com") is False

    def test_partial_ip_is_not_ip(self):
        assert _is_ip_address("192.168.1") is False

    def test_ipv6_stripped_literal(self):
        assert _is_ip_address("::1") is True

    def test_ipv6_full_address(self):
        assert _is_ip_address("2001:db8::1") is True

    def test_empty_string_is_not_ip(self):
        assert _is_ip_address("") is False

    def test_domain_starting_with_numbers_not_ip(self):
        assert _is_ip_address("123abc.com") is False


class TestCountSuspiciousKeywords:
    def test_no_keywords(self):
        assert _count_suspicious_keywords("https://example.com/news") == 0

    def test_single_keyword(self):
        assert _count_suspicious_keywords("https://example.com/login") == 1

    def test_multiple_keywords(self):
        count = _count_suspicious_keywords("https://example.com/verify-account-password")
        assert count >= 3

    def test_case_insensitive_via_lower_input(self):
        assert _count_suspicious_keywords("https://paypal-secure.xyz/update") >= 3


class TestFeatureDictShape:
    def test_returns_dict(self):
        result = extract_url_features("https://example.com")
        assert isinstance(result, dict)

    def test_contains_all_expected_keys(self):
        result = extract_url_features("https://example.com")
        assert result.keys() == EXPECTED_KEYS

    def test_all_values_are_numeric(self):
        result = extract_url_features("https://example.com")
        for key, val in result.items():
            assert isinstance(val, (int, float)), f"{key} has non-numeric value {val!r}"

    def test_no_none_values(self):
        result = extract_url_features("https://example.com")
        assert all(v is not None for v in result.values())


class TestOrdinaryHttpsUrl:
    URL = "https://www.example.com/path/to/page?q=hello"

    def setup_method(self):
        self.features = extract_url_features(self.URL)

    def test_url_length(self):
        assert self.features["url_length"] == len(self.URL)

    def test_is_https(self):
        assert self.features["is_https"] == 1

    def test_is_not_ip(self):
        assert self.features["is_ip_host"] == 0

    def test_no_non_std_port(self):
        assert self.features["has_non_std_port"] == 0

    def test_no_suspicious_tld(self):
        assert self.features["has_suspicious_tld"] == 0

    def test_hostname_length_positive(self):
        assert self.features["hostname_length"] > 0

    def test_path_length_positive(self):
        assert self.features["path_length"] > 0

    def test_query_length_positive(self):
        assert self.features["query_length"] > 0

    def test_dot_count_positive(self):
        assert self.features["dot_count"] > 0


class TestOrdinaryHttpUrl:
    def test_is_https_zero(self):
        features = extract_url_features("http://example.com")
        assert features["is_https"] == 0

    def test_no_query_gives_zero_query_length(self):
        features = extract_url_features("http://example.com/page")
        assert features["query_length"] == 0


class TestMinimalUrl:
    def test_bare_domain_path_length(self):
        features = extract_url_features("https://example.com")
        assert features["path_length"] >= 0

    def test_bare_domain_query_length_zero(self):
        features = extract_url_features("https://example.com")
        assert features["query_length"] == 0


class TestSuspiciousUrl:
    URL = "http://paypal-secure-login.xyz/verify-account?update=password"

    def setup_method(self):
        self.features = extract_url_features(self.URL)

    def test_has_suspicious_tld(self):
        assert self.features["has_suspicious_tld"] == 1

    def test_is_not_https(self):
        assert self.features["is_https"] == 0

    def test_suspicious_kw_count_high(self):
        assert self.features["suspicious_kw_count"] >= 5


class TestIpv4Host:
    def test_is_ip_host_set(self):
        features = extract_url_features("http://192.168.0.1/admin")
        assert features["is_ip_host"] == 1

    def test_hostname_length_matches_ip(self):
        features = extract_url_features("http://192.168.0.1/admin")
        assert features["hostname_length"] == len("192.168.0.1")


class TestIpv6Host:
    def test_is_ip_host_set_for_ipv6(self):
        features = extract_url_features("http://[::1]/path")
        assert features["is_ip_host"] == 1


class TestNonStandardPort:
    def test_port_8080_flagged(self):
        features = extract_url_features("http://example.com:8080/login")
        assert features["has_non_std_port"] == 1

    def test_port_443_not_flagged(self):
        features = extract_url_features("https://example.com:443/page")
        assert features["has_non_std_port"] == 0

    def test_port_80_not_flagged(self):
        features = extract_url_features("http://example.com:80/page")
        assert features["has_non_std_port"] == 0

    def test_port_4443_flagged(self):
        features = extract_url_features("https://example.com:4443/secure")
        assert features["has_non_std_port"] == 1


class TestSuspiciousTld:
    @pytest.mark.parametrize("tld", ["tk", "ml", "ga", "cf", "gq", "xyz", "top"])
    def test_known_bad_tld_flagged(self, tld):
        features = extract_url_features(f"http://example.{tld}/page")
        assert features["has_suspicious_tld"] == 1

    def test_com_not_flagged(self):
        features = extract_url_features("https://example.com/page")
        assert features["has_suspicious_tld"] == 0

    def test_org_not_flagged(self):
        features = extract_url_features("https://example.org/page")
        assert features["has_suspicious_tld"] == 0


class TestSubdomainDepth:
    def test_no_subdomain_gives_zero(self):
        features = extract_url_features("https://example.com/page")
        assert features["subdomain_depth"] == 0

    def test_www_gives_depth_one(self):
        features = extract_url_features("https://www.example.com/page")
        assert features["subdomain_depth"] == 1

    def test_deep_subdomain(self):
        features = extract_url_features("https://a.b.c.example.com/page")
        assert features["subdomain_depth"] == 3


class TestDomainEntropy:
    def test_entropy_non_negative(self):
        features = extract_url_features("https://example.com")
        assert features["domain_entropy"] >= 0.0

    def test_high_entropy_domain(self):
        high = extract_url_features("https://xk3z9mq7.com/page")
        low  = extract_url_features("https://aaa.com/page")
        assert high["domain_entropy"] > low["domain_entropy"]

    def test_entropy_is_float(self):
        features = extract_url_features("https://example.com")
        assert isinstance(features["domain_entropy"], float)


class TestInvalidInputs:
    def test_none_raises_valueerror(self):
        with pytest.raises(ValueError):
            extract_url_features(None)

    def test_empty_string_raises_valueerror(self):
        with pytest.raises(ValueError):
            extract_url_features("")

    def test_whitespace_only_raises_valueerror(self):
        with pytest.raises(ValueError):
            extract_url_features("   ")

    def test_integer_raises_valueerror(self):
        with pytest.raises(ValueError):
            extract_url_features(42)

    def test_no_scheme_raises_valueerror(self):
        with pytest.raises(ValueError):
            extract_url_features("example.com/page")

    def test_ftp_scheme_raises_valueerror(self):
        with pytest.raises(ValueError):
            extract_url_features("ftp://example.com/file")

    def test_oversized_url_raises_valueerror(self):
        long_url = "https://example.com/" + "a" * URL_MAX_LENGTH
        with pytest.raises(ValueError):
            extract_url_features(long_url)


class TestLevenshtein:
    def test_identical_strings_is_zero(self):
        assert _levenshtein("amazon", "amazon") == 0

    def test_one_substitution(self):
        assert _levenshtein("amaz0n", "amazon") == 1

    def test_one_deletion(self):
        assert _levenshtein("amazn", "amazon") == 1

    def test_one_insertion(self):
        assert _levenshtein("amazons", "amazon") == 1

    def test_transposition_costs_two(self):
        assert _levenshtein("arnazon", "amazon") == 2

    def test_empty_string_vs_other(self):
        assert _levenshtein("", "abc") == 3
        assert _levenshtein("abc", "") == 3

    def test_completely_different(self):
        d = _levenshtein("zzzzz", "amazon")
        assert d > 3

    def test_symmetry(self):
        assert _levenshtein("paypal", "paypol") == _levenshtein("paypol", "paypal")


class TestBrandMinEditDistance:
    def test_exact_brand_is_zero(self):
        assert _brand_min_edit_distance("amazon") == 0.0

    def test_digit_substitution_close_to_brand(self):
        d = _brand_min_edit_distance("amaz0n")
        assert d <= 2.0

    def test_transposition_close_to_brand(self):
        d = _brand_min_edit_distance("arnazon")
        assert d <= 3.0

    def test_unrelated_domain_far_from_brands(self):
        d = _brand_min_edit_distance("wikipedia")
        assert d >= 4.0

    def test_empty_domain_returns_99(self):
        assert _brand_min_edit_distance("") == 99.0

    def test_returns_float(self):
        assert isinstance(_brand_min_edit_distance("amazon"), float)


class TestBrandInSubdomainHelper:
    def test_brand_in_subdomain_not_in_domain(self):
        assert _brand_in_subdomain("amazon", "attacker") is True

    def test_brand_in_both_sub_and_domain_is_false(self):
        assert _brand_in_subdomain("www", "amazon") is False

    def test_no_subdomain_is_false(self):
        assert _brand_in_subdomain("", "amazon") is False

    def test_unrelated_subdomain_is_false(self):
        assert _brand_in_subdomain("www", "example") is False

    def test_partial_brand_match_in_subdomain(self):
        assert _brand_in_subdomain("amazon-secure", "attacker") is True


class TestHasDigitSubstitution:
    def test_amaz0n_flagged(self):
        assert _has_digit_substitution("amaz0n") is True

    def test_paypa1_flagged(self):
        assert _has_digit_substitution("paypa1") is True

    def test_real_amazon_not_flagged(self):
        assert _has_digit_substitution("amazon") is False

    def test_office365_not_flagged(self):
        assert _has_digit_substitution("office365") is False

    def test_numeric_only_domain_not_flagged(self):
        assert _has_digit_substitution("12345") is False

    def test_empty_not_flagged(self):
        assert _has_digit_substitution("") is False


class TestHasHyphenBrand:
    def test_paypal_hyphen_secure_flagged(self):
        assert _has_hyphen_brand("paypal-secure") is True

    def test_amazon_login_flagged(self):
        assert _has_hyphen_brand("amazon-login") is True

    def test_amazon_alone_not_flagged(self):
        assert _has_hyphen_brand("amazon") is False

    def test_example_not_flagged(self):
        assert _has_hyphen_brand("example-test") is False

    def test_no_hyphen_not_flagged(self):
        assert _has_hyphen_brand("paypal") is False

    def test_empty_not_flagged(self):
        assert _has_hyphen_brand("") is False


class TestLongestConsecutiveDigits:
    def test_no_digits_is_zero(self):
        assert _longest_consecutive_digits("amazon") == 0

    def test_single_digit(self):
        assert _longest_consecutive_digits("amaz0n") == 1

    def test_multi_digit_run(self):
        assert _longest_consecutive_digits("xk3z9mq7") == 1

    def test_four_digit_run(self):
        assert _longest_consecutive_digits("abc1234def") == 4

    def test_empty_is_zero(self):
        assert _longest_consecutive_digits("") == 0


class TestHasRedirectParam:
    def test_next_param_detected(self):
        assert _has_redirect_param("next=http://evil.com") is True

    def test_redirect_param_detected(self):
        assert _has_redirect_param("redirect=http://evil.com") is True

    def test_url_param_detected(self):
        assert _has_redirect_param("url=http://evil.com") is True

    def test_goto_param_detected(self):
        assert _has_redirect_param("goto=http://evil.com") is True

    def test_return_url_detected(self):
        assert _has_redirect_param("return_url=http://evil.com") is True

    def test_regular_query_not_detected(self):
        assert _has_redirect_param("q=search+term&page=2") is False

    def test_empty_query_not_detected(self):
        assert _has_redirect_param("") is False

    def test_case_insensitive(self):
        assert _has_redirect_param("REDIRECT=http://evil.com") is True


class TestHasUserinfo:
    def test_user_at_host_detected(self):
        assert _has_userinfo("user@example.com") is True

    def test_user_pass_at_host_detected(self):
        assert _has_userinfo("user:pass@example.com") is True

    def test_plain_hostname_not_detected(self):
        assert _has_userinfo("example.com") is False

    def test_empty_not_detected(self):
        assert _has_userinfo("") is False


class TestPathHasBrand:
    def test_amazon_in_path_detected(self):
        assert _path_has_brand("/amazon/account") is True

    def test_paypal_in_path_detected(self):
        assert _path_has_brand("/paypal/login") is True

    def test_generic_path_not_detected(self):
        assert _path_has_brand("/help/faq") is False

    def test_empty_path_not_detected(self):
        assert _path_has_brand("") is False


class TestTldLabelCount:
    def test_com_is_one(self):
        assert _tld_label_count("com") == 1

    def test_co_uk_is_two(self):
        assert _tld_label_count("co.uk") == 2

    def test_com_au_is_two(self):
        assert _tld_label_count("com.au") == 2

    def test_empty_is_zero(self):
        assert _tld_label_count("") == 0

class TestHostnameLabelCount:
    def test_bare_domain(self):
        f = extract_url_features("https://example.com")
        assert f["hostname_label_count"] == 2

    def test_with_www(self):
        f = extract_url_features("https://www.example.com")
        assert f["hostname_label_count"] == 3

    def test_deep_subdomain(self):
        f = extract_url_features("https://a.b.c.example.com")
        assert f["hostname_label_count"] == 5

    def test_ip_host_gives_four_labels(self):
        f = extract_url_features("http://192.168.0.1/page")
        assert f["hostname_label_count"] == 4


class TestPathDepth:
    def test_no_path(self):
        f = extract_url_features("https://example.com")
        assert f["path_depth"] == 0

    def test_single_segment(self):
        f = extract_url_features("https://example.com/page")
        assert f["path_depth"] == 1

    def test_three_segments(self):
        f = extract_url_features("https://example.com/a/b/c")
        assert f["path_depth"] == 3


class TestHasAtInAuthority:
    def test_no_at_sign(self):
        f = extract_url_features("https://example.com/page")
        assert f["has_at_in_authority"] == 0

    def test_at_sign_in_netloc(self):
        f = extract_url_features("http://user@example.com/page")
        assert f["has_at_in_authority"] == 1


class TestHasEncodedChars:
    def test_no_encoding(self):
        f = extract_url_features("https://example.com/page")
        assert f["has_encoded_chars"] == 0

    def test_encoded_path(self):
        f = extract_url_features("https://example.com/%61%6D%61%7A%6F%6E/page")
        assert f["has_encoded_chars"] == 1

    def test_encoded_query(self):
        f = extract_url_features("https://example.com/page?q=%73%65%63%72%65%74")
        assert f["has_encoded_chars"] == 1


class TestHasRedirectParamFeature:
    def test_no_redirect_param(self):
        f = extract_url_features("https://example.com/search?q=test")
        assert f["has_redirect_param"] == 0

    def test_next_param_flagged(self):
        f = extract_url_features("https://example.com/login?next=http%3A//evil.com")
        assert f["has_redirect_param"] == 1

    def test_redirect_param_flagged(self):
        f = extract_url_features("https://example.com/?redirect=http://evil.com")
        assert f["has_redirect_param"] == 1

    def test_url_param_flagged(self):
        f = extract_url_features("https://example.com/?url=http://other.com")
        assert f["has_redirect_param"] == 1


class TestIsIdn:
    def test_no_idn(self):
        f = extract_url_features("https://example.com")
        assert f["is_idn"] == 0

    def test_idn_punycode_detected(self):
        f = extract_url_features("http://xn--pple-43d.com/account/login")
        assert f["is_idn"] == 1

    def test_subdomain_idn_detected(self):
        f = extract_url_features("https://xn--mazon-yua.example.com/page")
        assert f["is_idn"] == 1


class TestBrandMinEditDistFeature:
    def test_real_amazon_com_is_zero(self):
        f = extract_url_features("https://www.amazon.com/product")
        assert f["brand_min_edit_dist"] == 0.0

    def test_amaz0n_is_close(self):
        f = extract_url_features("https://www.amaz0n.com/")
        assert f["brand_min_edit_dist"] <= 2.0

    def test_arnazon_is_close(self):
        f = extract_url_features("https://arnazon.com/login")
        assert f["brand_min_edit_dist"] <= 3.0

    def test_unrelated_domain_is_far(self):
        f = extract_url_features("https://wikipedia.org/wiki/Phishing")
        assert f["brand_min_edit_dist"] >= 4.0

    def test_ip_host_edit_dist_is_numeric(self):

        f = extract_url_features("http://192.168.0.1/page")
        assert isinstance(f["brand_min_edit_dist"], float)
        assert f["brand_min_edit_dist"] >= 0.0

    def test_edit_dist_is_float(self):
        f = extract_url_features("https://example.com")
        assert isinstance(f["brand_min_edit_dist"], float)


class TestBrandInSubdomainFeature:
    def test_amazon_in_subdomain_evil_domain(self):
        f = extract_url_features("https://amazon.attacker.com/login")
        assert f["brand_in_subdomain"] == 1

    def test_real_amazon_com_not_flagged(self):
        f = extract_url_features("https://www.amazon.com/page")
        assert f["brand_in_subdomain"] == 0

    def test_no_subdomain_not_flagged(self):
        f = extract_url_features("https://example.com/page")
        assert f["brand_in_subdomain"] == 0

    def test_unrelated_subdomain_not_flagged(self):
        f = extract_url_features("https://news.bbc.co.uk/page")
        assert f["brand_in_subdomain"] == 0


class TestHasDigitSubInDomainFeature:
    def test_amaz0n_flagged(self):
        f = extract_url_features("https://www.amaz0n.com/")
        assert f["has_digit_sub_in_domain"] == 1

    def test_paypa1_flagged(self):
        f = extract_url_features("http://paypa1.com/login")
        assert f["has_digit_sub_in_domain"] == 1

    def test_real_amazon_not_flagged(self):
        f = extract_url_features("https://www.amazon.com/")
        assert f["has_digit_sub_in_domain"] == 0

    def test_numeric_only_domain_not_flagged(self):
        f = extract_url_features("https://123.example.com/")
        assert f["has_digit_sub_in_domain"] == 0


class TestHasHyphenBrandFeature:
    def test_paypal_secure_flagged(self):
        f = extract_url_features("http://paypal-secure.com/login")
        assert f["has_hyphen_brand"] == 1

    def test_amazon_login_flagged(self):
        f = extract_url_features("http://amazon-login.net/account")
        assert f["has_hyphen_brand"] == 1

    def test_real_amazon_not_flagged(self):
        f = extract_url_features("https://www.amazon.com/page")
        assert f["has_hyphen_brand"] == 0

    def test_unrelated_hyphen_not_flagged(self):
        f = extract_url_features("https://my-website.com/page")
        assert f["has_hyphen_brand"] == 0


class TestConsecutiveDigitsFeature:
    def test_no_digits(self):
        f = extract_url_features("https://example.com")
        assert f["consecutive_digits"] == 0

    def test_single_digit_substitution(self):
        f = extract_url_features("https://amaz0n.com")
        assert f["consecutive_digits"] == 1

    def test_long_digit_run(self):
        f = extract_url_features("https://xk123456.com/page")
        assert f["consecutive_digits"] == 6


class TestHasUserinfoFeature:
    def test_no_userinfo(self):
        f = extract_url_features("https://example.com/page")
        assert f["has_userinfo"] == 0

    def test_userinfo_detected(self):
        f = extract_url_features("http://user@example.com/page")
        assert f["has_userinfo"] == 1


class TestPathHasBrandFeature:
    def test_amazon_in_path(self):
        f = extract_url_features("https://attacker.com/amazon/login")
        assert f["path_has_brand"] == 1

    def test_paypal_in_path(self):
        f = extract_url_features("https://evil.xyz/paypal/account")
        assert f["path_has_brand"] == 1

    def test_generic_path_no_brand(self):
        f = extract_url_features("https://example.com/help/faq")
        assert f["path_has_brand"] == 0

    def test_real_amazon_path_not_flagged_on_path_alone(self):
        f = extract_url_features("https://www.amazon.com/gp/help/customer")
        assert f["path_has_brand"] == 0


class TestTldLabelCountFeature:
    def test_com_is_one(self):
        f = extract_url_features("https://example.com/page")
        assert f["tld_label_count"] == 1

    def test_co_uk_is_two(self):
        f = extract_url_features("https://www.bbc.co.uk/news")
        assert f["tld_label_count"] == 2

    def test_ip_host_is_zero(self):
        f = extract_url_features("http://192.168.0.1/page")
        assert f["tld_label_count"] == 0

class TestAmazOn0Regression:

    URL = "https://www.amaz0n.com/"

    def setup_method(self):
        self.f = extract_url_features(self.URL)

    def test_digit_substitution_flagged(self):
        assert self.f["has_digit_sub_in_domain"] == 1

    def test_edit_distance_small(self):
        assert self.f["brand_min_edit_dist"] <= 2.0

    def test_is_https_does_not_mask_other_signals(self):
        assert self.f["is_https"] == 1
        assert self.f["has_digit_sub_in_domain"] == 1

    def test_has_suspicious_tld_is_zero(self):
        assert self.f["has_suspicious_tld"] == 0


class TestTyposquattingVariants:
    @pytest.mark.parametrize("url,expected_max_dist", [
        ("https://amaz0n.com/login",     2),
        ("https://arnazon.com/login",    3),
        ("https://amazoon.com/login",    2),
        ("https://amazn.com/login",      2),
    ])
    def test_lookalike_has_small_edit_distance(self, url, expected_max_dist):
        f = extract_url_features(url)
        assert f["brand_min_edit_dist"] <= expected_max_dist

    @pytest.mark.parametrize("url", [
        "https://paypal-secure.com/login",
        "https://microsoft-support.net/update",
        "https://apple-id-help.com/unlock",
    ])
    def test_hyphen_brand_pattern_flagged(self, url):
        f = extract_url_features(url)
        assert f["has_hyphen_brand"] == 1

    def test_brand_in_subdomain_amazon_attacker(self):
        f = extract_url_features("https://amazon.evildomain.com/account")
        assert f["brand_in_subdomain"] == 1

    def test_idn_apple_lookalike(self):
        f = extract_url_features("http://xn--pple-43d.com/account/login")
        assert f["is_idn"] == 1


class TestLegitimateHardNegatives:

    def test_google_signin_no_digit_sub(self):
        f = extract_url_features("https://accounts.google.com/signin/")
        assert f["has_digit_sub_in_domain"] == 0

    def test_github_login_no_digit_sub(self):
        f = extract_url_features("https://github.com/login")
        assert f["has_digit_sub_in_domain"] == 0

    def test_amazon_product_page_no_hyphen_brand(self):
        f = extract_url_features("https://www.amazon.com/dp/B08N5WRWNW")
        assert f["has_hyphen_brand"] == 0

    def test_amazon_product_page_no_digit_sub(self):
        f = extract_url_features("https://www.amazon.com/dp/B08N5WRWNW")
        assert f["has_digit_sub_in_domain"] == 0

    def test_google_no_brand_in_subdomain(self):
        f = extract_url_features("https://accounts.google.com/signin/")
        assert f["brand_in_subdomain"] == 0

    def test_wikipedia_far_from_brands(self):
        f = extract_url_features("https://en.wikipedia.org/wiki/Phishing")
        assert f["brand_min_edit_dist"] >= 4.0

    def test_legitimate_long_path_no_digit_sub(self):
        f = extract_url_features(
            "https://docs.python.org/3/library/urllib.parse.html"
        )
        assert f["has_digit_sub_in_domain"] == 0

    def test_legitimate_query_params_no_redirect(self):
        f = extract_url_features(
            "https://www.youtube.com/results?search_query=python"
        )
        assert f["has_redirect_param"] == 0

    def test_bbc_co_uk_tld_label_count(self):
        f = extract_url_features("https://www.bbc.co.uk/news/technology")
        assert f["tld_label_count"] == 2

    def test_microsoft_support_no_redirect_param(self):
        f = extract_url_features("https://support.microsoft.com/en-us/windows")
        assert f["has_redirect_param"] == 0


class TestOpenRedirectPattern:
    def test_google_open_redirect_next_param(self):
        f = extract_url_features(
            "https://accounts.google.com/ServiceLogin?continue=http://evil.com"
        )
        assert f["has_redirect_param"] == 1

    def test_redirect_uri_param(self):
        f = extract_url_features(
            "https://auth.example.com/oauth?redirect_uri=http://attacker.com"
        )
        assert f["has_redirect_param"] == 1


class TestEncodedUrls:
    def test_encoded_brand_in_path_flagged(self):
        f = extract_url_features(
            "https://attacker.com/%61%6D%61%7A%6F%6E/signin"
        )
        assert f["has_encoded_chars"] == 1

    def test_percent_encoding_in_query(self):
        f = extract_url_features(
            "https://example.com/login?next=%2Fdashboard"
        )
        assert f["has_encoded_chars"] == 1


class TestKnownBrandsConstant:
    def test_brands_are_lowercase(self):
        for brand in _KNOWN_BRANDS:
            assert brand == brand.lower(), f"Brand not lowercase: {brand!r}"

    def test_brands_are_non_empty(self):
        for brand in _KNOWN_BRANDS:
            assert len(brand) > 0

    def test_no_tld_in_brand_names(self):
        for brand in _KNOWN_BRANDS:
            assert "." not in brand, f"Brand contains dot: {brand!r}"
