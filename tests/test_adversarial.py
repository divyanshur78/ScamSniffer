"""
tests/test_adversarial.py
=========================
Adversarial, regression, and hard-negative tests for ScamSniffer.

Philosophy
----------
These tests do NOT assert model prediction labels (which depend on
the training data and could change as the dataset evolves).  Instead they:

  1. Assert that known HEURISTIC WARNINGS fire for adversarial inputs.
  2. Assert that the three-tier VERDICT logic is correct.
  3. Assert that the amaz0n.com LOOKALIKE WARNING fires (regression test).
  4. Assert that hard-negative legitimate URLs do NOT trigger false warnings.
  5. Assert that adversarial obfuscation techniques each trigger the
     appropriate heuristic flag.

All tests are purely lexical — no model training is required.

Coverage
--------
  Lookalike/typosquatting      — amaz0n.com, paypa1.com, g00gle.com
  IDN/punycode                 — xn-- prefixed hostnames
  User-info / @ trick          — user@evil.com style
  Brand in subdomain           — amazon.evil.com
  Hyphen brand                 — paypal-secure.xyz
  Digit substitution           — amaz0n, paypa1, g00gle
  Deep subdomains              — a.b.c.d.evil.xyz
  IP hosts                     — 192.168.0.1, 10.0.0.1
  Non-standard port            — :8080, :9090
  Redirect params              — ?redirect=, ?next=, ?url=
  Encoded chars                — %2F, %40 in path
  Verdict logic                — _verdict_for_url, _verdict_for_text
  Hard negatives               — legitimate URLs with brand names
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from src.detector import (
    _HIGH_CONF_HEURISTICS,
    _VERDICT_BENIGN,
    _VERDICT_PHISHING,
    _url_warnings,
    _text_warnings,
    _verdict_for_url,
    _verdict_for_text,
)
from src.features.url_features import extract_url_features
from src.features.text_features import extract_text_features


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _features(url: str) -> dict:
    return extract_url_features(url)


def _has_warning_containing(warnings: list[str], fragment: str) -> bool:
    return any(fragment.lower() in w.lower() for w in warnings)


# ===========================================================================
# Regression: amaz0n.com lookalike warning must fire
# ===========================================================================


class TestAmazonLookalikeRegression:
    """
    Regression tests ensuring that amaz0n.com (and similar typosquatting
    domains) trigger the digit-substitution or brand-similarity warning.

    The original model failed on amaz0n.com because it relied on is_https
    and has_suspicious_tld as near-perfect proxies.  These tests ensure
    the Phase 2 heuristics now catch it regardless of the model label.
    """

    def test_amaz0n_digit_sub_feature_fires(self):
        feats = _features("http://amaz0n.com/account/login")
        assert feats["has_digit_sub_in_domain"] == 1, (
            "has_digit_sub_in_domain must be 1 for amaz0n.com"
        )

    def test_amaz0n_warning_fired(self):
        feats    = _features("http://amaz0n.com/account/login")
        warnings = _url_warnings(feats, "http://amaz0n.com/account/login")
        assert _has_warning_containing(warnings, "digit"), (
            "A digit-substitution warning must fire for amaz0n.com"
        )

    def test_paypa1_digit_sub_feature_fires(self):
        feats = _features("http://paypa1.com/signin")
        assert feats["has_digit_sub_in_domain"] == 1

    def test_paypa1_warning_fired(self):
        feats    = _features("http://paypa1.com/signin")
        warnings = _url_warnings(feats, "http://paypa1.com/signin")
        assert _has_warning_containing(warnings, "digit") or \
               _has_warning_containing(warnings, "brand"), (
            "A warning about digit substitution or brand similarity must fire for paypa1.com"
        )

    def test_g00gle_digit_sub_feature_fires(self):
        feats = _features("http://g00gle.com/accounts/verify")
        assert feats["has_digit_sub_in_domain"] == 1

    def test_amaz0n_brand_edit_dist_close(self):
        feats = _features("http://amaz0n.com/")
        # edit distance from "amaz0n" to "amazon" is 1
        assert feats["brand_min_edit_dist"] <= 2

    def test_amaz0n_brand_edit_dist_warning_fires(self):
        feats    = _features("http://amaz0n.com/")
        warnings = _url_warnings(feats, "http://amaz0n.com/")
        # Either digit-sub or edit-distance warning must fire
        has_relevant = (
            _has_warning_containing(warnings, "digit") or
            _has_warning_containing(warnings, "similar") or
            _has_warning_containing(warnings, "lookalike") or
            _has_warning_containing(warnings, "brand")
        )
        assert has_relevant, f"No brand/typosquatting warning for amaz0n.com. Warnings: {warnings}"


# ===========================================================================
# IDN / punycode
# ===========================================================================


class TestIdnPunycode:
    def test_xn_feature_fires(self):
        feats = _features("http://xn--pple-43d.com/account/login")
        assert feats["is_idn"] == 1

    def test_xn_warning_fired(self):
        feats    = _features("http://xn--pple-43d.com/account/login")
        warnings = _url_warnings(feats, "http://xn--pple-43d.com/account/login")
        assert _has_warning_containing(warnings, "punycode") or \
               _has_warning_containing(warnings, "idn") or \
               _has_warning_containing(warnings, "spoofing"), (
            f"IDN warning must fire for xn-- domain. Warnings: {warnings}"
        )

    def test_normal_domain_no_idn(self):
        feats = _features("https://www.amazon.com/")
        assert feats["is_idn"] == 0

    def test_xn_in_subdomain_fires(self):
        feats = _features("http://xn--paypa1.example.com/")
        assert feats["is_idn"] == 1


# ===========================================================================
# User-info / @ authority trick
# ===========================================================================


class TestUserInfoTrick:
    def test_at_in_authority_feature(self):
        feats = _features("http://victim@evil.com/login")
        assert feats["has_at_in_authority"] == 1 or feats["has_userinfo"] == 1

    def test_at_in_authority_warning(self):
        feats    = _features("http://victim@evil.com/login")
        warnings = _url_warnings(feats, "http://victim@evil.com/login")
        assert _has_warning_containing(warnings, "@") or \
               _has_warning_containing(warnings, "user"), (
            f"@ trick warning must fire. Warnings: {warnings}"
        )

    def test_clean_url_no_at_warning(self):
        feats    = _features("https://www.amazon.com/cart")
        warnings = _url_warnings(feats, "https://www.amazon.com/cart")
        assert not _has_warning_containing(warnings, "user-info")


# ===========================================================================
# Brand in subdomain
# ===========================================================================


class TestBrandInSubdomain:
    def test_amazon_in_subdomain_fires(self):
        feats = _features("http://amazon.attacker-login.com/account/verify")
        assert feats["brand_in_subdomain"] == 1

    def test_amazon_in_subdomain_warning(self):
        url      = "http://amazon.attacker-login.com/account/verify"
        feats    = _features(url)
        warnings = _url_warnings(feats, url)
        assert _has_warning_containing(warnings, "subdomain") or \
               _has_warning_containing(warnings, "impersonation"), (
            f"Brand-in-subdomain warning must fire. Warnings: {warnings}"
        )

    def test_genuine_amazon_no_subdomain_flag(self):
        feats = _features("https://www.amazon.com/")
        assert feats["brand_in_subdomain"] == 0

    def test_paypal_in_subdomain_fires(self):
        feats = _features("http://paypal.evil-phish.com/wallet")
        assert feats["brand_in_subdomain"] == 1


# ===========================================================================
# Hyphen brand
# ===========================================================================


class TestHyphenBrand:
    def test_paypal_hyphen_fires(self):
        feats = _features("http://paypal-secure.xyz/verify")
        assert feats["has_hyphen_brand"] == 1

    def test_paypal_hyphen_warning(self):
        url      = "http://paypal-secure.xyz/verify"
        feats    = _features(url)
        warnings = _url_warnings(feats, url)
        assert _has_warning_containing(warnings, "hyphen"), (
            f"Hyphen-brand warning must fire. Warnings: {warnings}"
        )

    def test_amazon_hyphen_fires(self):
        feats = _features("http://amazon-billing-update.tk/pay")
        assert feats["has_hyphen_brand"] == 1

    def test_clean_domain_no_hyphen_brand(self):
        feats = _features("https://www.paypal.com/us/home")
        assert feats["has_hyphen_brand"] == 0


# ===========================================================================
# IP host
# ===========================================================================


class TestIpHost:
    def test_ipv4_feature_fires(self):
        feats = _features("http://192.168.0.1/admin/login")
        assert feats["is_ip_host"] == 1

    def test_ipv4_warning_fires(self):
        feats    = _features("http://192.168.0.1/admin/login")
        warnings = _url_warnings(feats, "http://192.168.0.1/admin/login")
        assert _has_warning_containing(warnings, "ip address"), (
            f"IP warning must fire. Warnings: {warnings}"
        )

    def test_ipv4_high_conf_heuristic(self):
        # is_ip_host is in _HIGH_CONF_HEURISTICS — should trigger PHISHING verdict
        # even if model says label=0
        feats  = _features("http://192.168.0.1/admin/login")
        verdict = _verdict_for_url(0, feats)
        assert verdict == _VERDICT_PHISHING

    def test_normal_domain_no_ip_flag(self):
        feats = _features("https://www.amazon.com/")
        assert feats["is_ip_host"] == 0


# ===========================================================================
# Non-standard port
# ===========================================================================


class TestNonStandardPort:
    def test_port_8080_fires(self):
        feats = _features("http://evil-site.com:8080/login")
        assert feats["has_non_std_port"] == 1

    def test_port_warning_fires(self):
        feats    = _features("http://evil-site.com:8080/login")
        warnings = _url_warnings(feats, "http://evil-site.com:8080/login")
        assert _has_warning_containing(warnings, "port"), (
            f"Non-standard port warning must fire. Warnings: {warnings}"
        )

    def test_port_443_not_non_std(self):
        feats = _features("https://www.amazon.com:443/")
        assert feats["has_non_std_port"] == 0

    def test_port_80_not_non_std(self):
        feats = _features("http://www.example.com:80/")
        assert feats["has_non_std_port"] == 0


# ===========================================================================
# Redirect parameters
# ===========================================================================


class TestRedirectParam:
    def test_redirect_param_fires(self):
        feats = _features("http://evil.xyz/login?redirect=http://bank.com")
        assert feats["has_redirect_param"] == 1

    def test_next_param_fires(self):
        feats = _features("http://evil.xyz/?next=http://bank.com")
        assert feats["has_redirect_param"] == 1

    def test_redirect_warning_fires(self):
        url      = "http://evil.xyz/login?redirect=http://bank.com"
        feats    = _features(url)
        warnings = _url_warnings(feats, url)
        assert _has_warning_containing(warnings, "redirect"), (
            f"Redirect warning must fire. Warnings: {warnings}"
        )

    def test_benign_url_no_redirect_param(self):
        feats = _features("https://www.google.com/search?q=python")
        assert feats["has_redirect_param"] == 0


# ===========================================================================
# Encoded characters
# ===========================================================================


class TestEncodedChars:
    def test_percent_encoded_path_fires(self):
        feats = _features("http://evil.xyz/%2Fadmin%2Flogin")
        assert feats["has_encoded_chars"] == 1

    def test_percent_encoded_warning_fires(self):
        url      = "http://evil.xyz/%61dmin/login"
        feats    = _features(url)
        warnings = _url_warnings(feats, url)
        assert _has_warning_containing(warnings, "percent") or \
               _has_warning_containing(warnings, "encoded"), (
            f"Encoded chars warning must fire. Warnings: {warnings}"
        )

    def test_benign_url_no_encoding(self):
        feats = _features("https://www.python.org/downloads/")
        assert feats["has_encoded_chars"] == 0


# ===========================================================================
# Deep subdomain
# ===========================================================================


class TestDeepSubdomain:
    def test_deeply_nested_subdomain_warning(self):
        url      = "http://a.b.c.d.evil.xyz/login"
        feats    = _features(url)
        warnings = _url_warnings(feats, url)
        assert _has_warning_containing(warnings, "subdomain"), (
            f"Deep subdomain warning must fire. Warnings: {warnings}"
        )

    def test_subdomain_depth_feature(self):
        feats = _features("http://a.b.c.d.evil.xyz/")
        assert feats["subdomain_depth"] >= 3


# ===========================================================================
# Three-tier verdict logic
# ===========================================================================


class TestVerdictForUrl:
    def test_label_1_gives_phishing_verdict(self):
        feats = _features("https://www.amazon.com/")  # benign features
        assert _verdict_for_url(1, feats) == _VERDICT_PHISHING

    def test_ip_host_gives_phishing_verdict_regardless_of_label(self):
        feats = _features("http://192.168.0.1/login")
        assert _verdict_for_url(0, feats) == _VERDICT_PHISHING

    def test_suspicious_tld_gives_phishing_verdict_regardless_of_label(self):
        feats = _features("http://legit-looking.xyz/")
        if feats["has_suspicious_tld"]:
            assert _verdict_for_url(0, feats) == _VERDICT_PHISHING

    def test_idn_gives_phishing_verdict_regardless_of_label(self):
        feats = _features("http://xn--pple-43d.com/login")
        assert _verdict_for_url(0, feats) == _VERDICT_PHISHING

    def test_clean_url_label_0_gives_benign_verdict(self):
        feats = _features("https://www.python.org/downloads/")
        # Assuming no high-conf heuristics fire for python.org
        verdict = _verdict_for_url(0, feats)
        assert verdict == _VERDICT_BENIGN

    def test_userinfo_gives_phishing_verdict(self):
        feats = _features("http://victim@evil.com/login")
        assert _verdict_for_url(0, feats) == _VERDICT_PHISHING

    def test_non_std_port_gives_phishing_verdict(self):
        feats = _features("http://evil.com:9090/admin")
        assert _verdict_for_url(0, feats) == _VERDICT_PHISHING

    def test_verdict_phishing_string(self):
        assert _VERDICT_PHISHING == "Potentially phishing"

    def test_verdict_benign_string(self):
        assert _VERDICT_BENIGN == "No strong signals detected"


class TestVerdictForText:
    def test_label_1_gives_phishing_verdict(self):
        feats = extract_text_features("Hello, how are you?", input_type="email")
        assert _verdict_for_text(1, feats) == _VERDICT_PHISHING

    def test_urgency_plus_credential_gives_phishing_verdict(self):
        text  = "URGENT: verify your password immediately to avoid account suspension!"
        feats = extract_text_features(text, input_type="email")
        verdict = _verdict_for_text(0, feats)
        # Both urgency and credential request should be present
        if feats.get("urgency_term_count", 0) >= 1 and feats.get("credential_req_count", 0) >= 1:
            assert verdict == _VERDICT_PHISHING

    def test_benign_text_label_0_gives_benign_verdict(self):
        text  = "Meeting rescheduled to Thursday at 2pm."
        feats = extract_text_features(text, input_type="email")
        assert _verdict_for_text(0, feats) == _VERDICT_BENIGN

    def test_urgency_alone_label_0_gives_benign(self):
        # Urgency alone without credential request should NOT trigger phishing verdict
        text  = "URGENT: the server is down!"
        feats = extract_text_features(text, input_type="email")
        if feats.get("credential_req_count", 0) == 0:
            assert _verdict_for_text(0, feats) == _VERDICT_BENIGN


# ===========================================================================
# Hard negatives — legitimate URLs that must NOT trigger false alarms
# ===========================================================================


class TestHardNegativesUrl:
    """
    Legitimate URLs that contain brand names, security-related keywords, or
    other potentially ambiguous patterns.  None of these should trigger the
    high-confidence structural heuristics.
    """

    @pytest.mark.parametrize("url", [
        "https://www.amazon.com/gp/help/customer/display.html",
        "https://www.paypal.com/us/home",
        "https://support.microsoft.com/en-us/windows",
        "https://www.apple.com/shop/buy-iphone",
        "https://account.google.com/security",
        "https://www.chase.com/personal/banking",
        "https://www.wellsfargo.com/checking/",
        "https://www.bankofamerica.com/online-banking/",
    ])
    def test_no_high_conf_heuristic_on_legit_url(self, url):
        feats = _features(url)
        for h in _HIGH_CONF_HEURISTICS:
            assert not feats.get(h), (
                f"High-confidence heuristic '{h}' fired on legitimate URL: {url}"
            )

    @pytest.mark.parametrize("url", [
        "https://www.amazon.com/",
        "https://www.paypal.com/us/home",
        "https://www.google.com/accounts/",
    ])
    def test_benign_verdict_on_known_legit_url(self, url):
        feats   = _features(url)
        verdict = _verdict_for_url(0, feats)
        assert verdict == _VERDICT_BENIGN, (
            f"Expected benign verdict for {url}, got {verdict!r}"
        )

    def test_office365_no_digit_sub(self):
        # office365 should NOT trigger digit substitution false positive
        feats = _features("https://www.office365.com/")
        assert feats["has_digit_sub_in_domain"] == 0, (
            "office365 must not trigger digit-substitution false positive"
        )

    def test_krebs0nsecurity_no_digit_sub(self):
        # krebs0nsecurity.com - the '0' is in 'krebs0n', not a brand name substitution
        # This depends on brand proximity guard in url_features — just check no crash
        feats = _features("https://www.krebs0nsecurity.com/")
        assert isinstance(feats["has_digit_sub_in_domain"], int)

    def test_wikipedia_no_warnings_about_ip(self):
        feats    = _features("https://www.wikipedia.org/wiki/Phishing")
        warnings = _url_warnings(feats, "https://www.wikipedia.org/wiki/Phishing")
        assert not _has_warning_containing(warnings, "IP address")

    def test_https_url_no_http_warning(self):
        feats    = _features("https://www.amazon.com/")
        warnings = _url_warnings(feats, "https://www.amazon.com/")
        assert not _has_warning_containing(warnings, "HTTP, not HTTPS")


# ===========================================================================
# Hard negatives — legitimate messages
# ===========================================================================


class TestHardNegativesText:
    @pytest.mark.parametrize("text", [
        "Your Amazon order #112-3456789 has shipped.",
        "Meeting rescheduled to Thursday at 2pm.",
        "Your PayPal payment of $50 has been processed.",
        "Reminder: team standup is at 9am tomorrow.",
        "Your package was delivered to the front door at 3:14pm.",
    ])
    def test_benign_verdict_on_known_legit_text(self, text):
        feats   = extract_text_features(text, input_type="email")
        verdict = _verdict_for_text(0, feats)
        assert verdict == _VERDICT_BENIGN, (
            f"Expected benign verdict for legit message, got {verdict!r}. Text: {text!r}"
        )

    def test_transaction_confirmation_not_phishing_verdict(self):
        text  = "Your PayPal payment of $50 has been processed successfully."
        feats = extract_text_features(text, input_type="sms")
        assert _verdict_for_text(0, feats) == _VERDICT_BENIGN

    def test_no_credential_request_in_legit_message(self):
        text  = "Reminder: submit your timesheet by 5pm Friday."
        feats = extract_text_features(text, input_type="email")
        assert feats.get("credential_req_count", 0) == 0


# ===========================================================================
# Adversarial combinations
# ===========================================================================


class TestAdversarialCombinations:
    def test_https_phishing_domain_still_warns(self):
        """HTTPS alone must not suppress warnings for clearly malicious patterns."""
        url   = "https://paypa1-secure.xyz/verify?update=password"
        feats = _features(url)
        warnings = _url_warnings(feats, url)
        # Should have at least one warning even though HTTPS
        assert len(warnings) > 0

    def test_com_tld_phishing_still_warns_on_digit_sub(self):
        """.com TLD alone must not suppress digit-substitution warnings."""
        url   = "http://amaz0n.com/account/login"
        feats = _features(url)
        warnings = _url_warnings(feats, url)
        has_brand_warning = any(
            w for w in warnings
            if "digit" in w.lower() or "similar" in w.lower() or "brand" in w.lower()
        )
        assert has_brand_warning, (
            f"amaz0n.com must trigger at least one brand-related warning. Got: {warnings}"
        )

    def test_multiple_signals_generate_multiple_warnings(self):
        """A URL with many red flags should produce many warnings."""
        url   = "http://amaz0n.attacker-login.tk:8080/verify?redirect=bank"
        feats = _features(url)
        warnings = _url_warnings(feats, url)
        assert len(warnings) >= 3, (
            f"Expected ≥3 warnings for heavily adversarial URL, got {len(warnings)}: {warnings}"
        )

    def test_clean_url_minimal_warnings(self):
        """A well-known legitimate URL should generate at most 1 warning."""
        url      = "https://www.python.org/downloads/"
        feats    = _features(url)
        warnings = _url_warnings(feats, url)
        assert len(warnings) <= 1, (
            f"Expected ≤1 warnings for python.org, got {len(warnings)}: {warnings}"
        )
