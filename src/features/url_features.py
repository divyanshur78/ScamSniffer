"""
src/features/url_features.py
============================
Deterministic numeric feature extraction from a URL string.

⚠️  This module NEVER fetches, opens, executes, or resolves any URL.
    All analysis is purely lexical / structural.

⚠️  Brand-similarity features are WARNING SIGNALS only.  A low edit distance
    to a known brand does not prove phishing; legitimate URLs containing brand
    names (e.g. brand press-releases, official support pages) may score
    similarly.  Always combine with other signals and human judgement.

Public API
----------
extract_url_features(url: str) -> dict[str, int | float]
    Validate the URL with src.validator and return a fixed-length feature dict.

Feature catalogue — original (12 features)
-------------------------------------------
url_length              : int   Total character length of the URL
hostname_length         : int   Length of the hostname component
path_length             : int   Length of the path component (excl. query/fragment)
query_length            : int   Length of the raw query string (excl. leading '?')
dot_count               : int   Number of '.' characters in the full URL
subdomain_depth         : int   Number of labels in the subdomain prefix
domain_entropy          : float Shannon entropy of the registered domain label (bits)
is_ip_host              : int   1 if hostname is an IPv4 or IPv6 literal
is_https                : int   1 if scheme is https (not a trust signal on its own)
suspicious_kw_count     : int   Count of suspicious keywords in the URL
has_non_std_port        : int   1 if explicit non-standard port (not 80/443)
has_suspicious_tld      : int   1 if TLD is in the known-abuse list

Feature catalogue — new (14 features)
--------------------------------------
hostname_label_count    : int   Total dot-separated labels in the full hostname
path_depth              : int   Number of '/' segments in the path (depth)
has_at_in_authority     : int   1 if '@' appears in the authority (user-info trick)
has_encoded_chars       : int   1 if '%xx' percent-encoding appears in path or query
has_redirect_param      : int   1 if a common redirect parameter name is present
is_idn                  : int   1 if the hostname contains an 'xn--' punycode label
brand_min_edit_dist     : float Levenshtein distance from the registered domain
                                 to the closest known brand name (0 = exact match)
brand_in_subdomain      : int   1 if a brand name appears in the subdomain but NOT
                                 in the registered domain (brand impersonation pattern)
has_digit_sub_in_domain : int   1 if the registered domain contains a digit that
                                 substitutes a visually similar letter (0→o, 1→l/i,
                                 3→e, 5→s, @→a)
has_hyphen_brand        : int   1 if the registered domain contains a brand name
                                 joined by a hyphen (e.g. 'paypal-secure')
consecutive_digits      : int   Length of the longest run of consecutive digits in
                                 the registered domain (DGA indicator)
has_userinfo            : int   1 if 'user:pass@' or 'user@' pattern in the raw URL
                                 authority section
path_has_brand          : int   1 if a brand name appears in the URL path
tld_label_count         : int   Number of dot-separated labels in the public suffix
                                 (e.g. 'co.uk' → 2; useful for multi-part TLDs)

Assumptions / edge-case handling
---------------------------------
- Input is passed through validate_url() first; malformed values raise ValueError.
- URLs with no registrable domain (bare IP, localhost) yield brand_min_edit_dist=99,
  brand_in_subdomain=0, has_digit_sub_in_domain=0, etc.
- IPv6 literals detected via colon presence in urlparse.hostname (brackets stripped).
- subdomain_depth is 0 when tldextract finds no subdomain component.
- Levenshtein distance is capped at 99 when no domain label is present.
- is_https must NOT be treated as a definitive legitimacy signal; phishers routinely
  obtain valid TLS certificates for malicious domains.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from urllib.parse import urlparse, unquote

import tldextract

from src.validator import validate_url

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_SUSPICIOUS_KEYWORDS: frozenset[str] = frozenset(
    {
        "login",
        "logon",
        "signin",
        "sign-in",
        "verify",
        "verification",
        "secure",
        "security",
        "account",
        "update",
        "confirm",
        "password",
        "passwd",
        "credential",
        "banking",
        "bank",
        "paypal",
        "ebay",
        "amazon",
        "apple",
        "microsoft",
        "google",
        "support",
        "helpdesk",
        "alert",
        "suspend",
        "suspended",
        "unlock",
        "wallet",
        "recover",
        "recovery",
        "invoice",
        "payment",
    }
)

# Ports that are standard for http (80) and https (443).
_STANDARD_PORTS: frozenset[int] = frozenset({80, 443})

# TLDs historically associated with free/abused registrations.
# NOTE: this list alone must not determine a phishing verdict.
_SUSPICIOUS_TLDS: frozenset[str] = frozenset(
    {
        "tk",
        "ml",
        "ga",
        "cf",
        "gq",     # former Freenom free TLDs
        "xyz",
        "top",
        "club",
        "online",
        "site",
        "website",
        "space",
        "live",
        "click",
        "link",
        "work",
        "pw",
        "cc",
        "su",
    }
)

# Well-known brand names used for edit-distance and brand-placement checks.
# These are registered domain labels only (no TLD, no www).
# NOTE: this list is a heuristic signal; it is intentionally conservative to
# reduce false positives.  Absence from this list does NOT mean a domain is safe.
_KNOWN_BRANDS: tuple[str, ...] = (
    "amazon",
    "apple",
    "microsoft",
    "google",
    "paypal",
    "ebay",
    "facebook",
    "instagram",
    "netflix",
    "dropbox",
    "linkedin",
    "twitter",
    "chase",
    "wellsfargo",
    "bankofamerica",
    "citibank",
    "hsbc",
    "dhl",
    "fedex",
    "ups",
    "usps",
)

# Common redirect/open-redirect query parameter names.
_REDIRECT_PARAMS: frozenset[str] = frozenset(
    {
        "url",
        "redirect",
        "redirect_uri",
        "redirect_url",
        "next",
        "goto",
        "return",
        "returnto",
        "return_url",
        "target",
        "dest",
        "destination",
        "forward",
        "continue",
        "redir",
    }
)

# Digit-to-letter homoglyph mappings commonly used in typosquatting.
# Maps a digit character to the letter(s) it visually resembles.
_DIGIT_SUB_MAP: dict[str, str] = {
    "0": "o",
    "1": "li",
    "3": "e",
    "4": "a",
    "5": "s",
    "6": "g",
    "8": "b",
}

# Regex: IPv4 address (no port).
_IPV4_RE: re.Pattern[str] = re.compile(
    r"^(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)$"
)

# Regex: percent-encoded characters (%xx where xx are hex digits).
_PCT_ENCODED_RE: re.Pattern[str] = re.compile(r"%[0-9A-Fa-f]{2}")

# Regex: user-info in authority  (anything before '@' in the netloc).
_USERINFO_RE: re.Pattern[str] = re.compile(r"^[^@/]+@")

# Regex: consecutive digit run.
_CONSECUTIVE_DIGITS_RE: re.Pattern[str] = re.compile(r"\d+")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _shannon_entropy(text: str) -> float:
    """Return the Shannon entropy (base-2 bits) of *text*.

    Returns 0.0 for an empty string.
    """
    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    return -sum(
        (c / length) * math.log2(c / length) for c in counts.values()
    )


def _is_ip_address(hostname: str) -> bool:
    """Return True if *hostname* is an IPv4 or IPv6 literal.

    ``urlparse`` strips the surrounding brackets from IPv6 literals before
    returning ``parsed.hostname``, so the value seen here is e.g. ``'::1'``
    rather than ``'[::1]'``.  We detect IPv6 by the presence of a colon.
    """
    if not hostname:
        return False
    if ":" in hostname:
        return True
    return bool(_IPV4_RE.match(hostname))


def _count_suspicious_keywords(url_lower: str) -> int:
    """Count how many distinct suspicious keywords appear in *url_lower*."""
    return sum(1 for kw in _SUSPICIOUS_KEYWORDS if kw in url_lower)


def _levenshtein(s: str, t: str) -> int:
    """Compute the Levenshtein edit distance between *s* and *t*.

    Pure-Python implementation — no external library required.
    Uses the standard single-row DP algorithm; O(min(|s|,|t|)) space.

    Returns the integer edit distance.
    """
    if s == t:
        return 0
    if not s:
        return len(t)
    if not t:
        return len(s)
    # Ensure s is the shorter string (saves memory).
    if len(s) > len(t):
        s, t = t, s
    prev = list(range(len(s) + 1))
    for j, tc in enumerate(t, 1):
        curr = [j]
        for i, sc in enumerate(s, 1):
            cost = 0 if sc == tc else 1
            curr.append(min(prev[i] + 1, curr[i - 1] + 1, prev[i - 1] + cost))
        prev = curr
    return prev[-1]


def _brand_min_edit_distance(domain_label: str) -> float:
    """Return the minimum Levenshtein distance between *domain_label* and
    any name in ``_KNOWN_BRANDS``.

    Returns 99 when *domain_label* is empty (no registrable domain found,
    e.g. bare IP or localhost).

    The distance is a continuous signal — callers decide thresholds.
    """
    if not domain_label:
        return 99.0
    dl = domain_label.lower()
    return float(min(_levenshtein(dl, brand) for brand in _KNOWN_BRANDS))


def _brand_in_subdomain(subdomain: str, domain_label: str) -> bool:
    """Return True if a brand name appears in the *subdomain* but the
    registered domain (*domain_label*) does not itself match that brand.

    This detects patterns like ``amazon.attacker.com`` (brand hijacked into
    the subdomain to mislead users) while not firing on ``www.amazon.com``
    (brand IS the registered domain).
    """
    if not subdomain:
        return False
    sub_lower    = subdomain.lower()
    domain_lower = domain_label.lower()
    for brand in _KNOWN_BRANDS:
        if brand in sub_lower and brand not in domain_lower:
            return True
    return False


def _has_digit_substitution(domain_label: str) -> bool:
    """Return True if *domain_label* contains a digit that substitutes a
    visually similar letter according to ``_DIGIT_SUB_MAP``.

    The check works by replacing each substitutable digit with its letter
    equivalent and testing whether the result is closer to a known brand.
    This avoids false positives on numeric-only domains (e.g. '365', '24x7').

    The check is: does the domain contain a suspicious digit AND does the
    letter-substituted version match (or closely resemble) a known brand?
    """
    if not domain_label:
        return False
    dl = domain_label.lower()
    # Only flag if there is at least one substitutable digit.
    sub_digits = [c for c in dl if c in _DIGIT_SUB_MAP]
    if not sub_digits:
        return False
    # Build the substituted string and check closeness to brands.
    substituted = dl
    for digit, letters in _DIGIT_SUB_MAP.items():
        substituted = substituted.replace(digit, letters[0])
    # Flag if the substituted form is within edit distance 2 of a brand.
    min_dist = _brand_min_edit_distance(substituted)
    return min_dist <= 2


def _has_hyphen_brand(domain_label: str) -> bool:
    """Return True if the registered domain contains a brand name separated
    by a hyphen from additional text (e.g. 'paypal-secure', 'amazon-login').

    Exact brand matches without hyphens are not flagged here (they would
    just be the real brand domain).
    """
    if not domain_label:
        return False
    dl = domain_label.lower()
    if "-" not in dl:
        return False
    parts = dl.split("-")
    for part in parts:
        for brand in _KNOWN_BRANDS:
            if part == brand:
                return True
    return False


def _longest_consecutive_digits(domain_label: str) -> int:
    """Return the length of the longest run of consecutive digits in
    *domain_label* (a DGA / random-domain indicator)."""
    if not domain_label:
        return 0
    matches = _CONSECUTIVE_DIGITS_RE.findall(domain_label)
    return max((len(m) for m in matches), default=0)


def _has_redirect_param(query: str) -> bool:
    """Return True if the query string contains a known redirect parameter."""
    if not query:
        return False
    q_lower = query.lower()
    # Parse key=value pairs loosely (handles both & and ; separators).
    for pair in re.split(r"[&;]", q_lower):
        key = pair.split("=", 1)[0].strip()
        if key in _REDIRECT_PARAMS:
            return True
    return False


def _has_userinfo(netloc: str) -> bool:
    """Return True if *netloc* (the raw authority from urlparse) contains
    a user-info component (``user@`` or ``user:pass@`` before the host)."""
    return bool(_USERINFO_RE.match(netloc))


def _path_has_brand(path: str) -> bool:
    """Return True if the URL path contains a known brand name as a token."""
    if not path:
        return False
    path_lower = path.lower()
    for brand in _KNOWN_BRANDS:
        if brand in path_lower:
            return True
    return False


def _tld_label_count(suffix: str) -> int:
    """Return the number of dot-separated labels in the public suffix
    (e.g. 'co.uk' → 2, 'com' → 1, '' → 0)."""
    if not suffix:
        return 0
    return len(suffix.split("."))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_url_features(url: object) -> dict[str, int | float]:
    """
    Extract a fixed set of numeric features from *url*.

    Parameters
    ----------
    url:
        Raw URL value.  Passed through ``validate_url()`` before processing —
        invalid inputs raise ``ValueError`` with a descriptive message.

    Returns
    -------
    dict[str, int | float]
        Feature dictionary with the keys listed in the module docstring.
        Values are always numeric (``int`` or ``float``); never ``None``.

    Raises
    ------
    ValueError
        Propagated from ``validate_url()`` for non-string, empty, oversized,
        or scheme-less inputs.

    Notes
    -----
    ``is_https`` is included as an informational feature but MUST NOT be used
    as a sole or dominant phishing signal.  Phishers routinely obtain valid
    TLS certificates.  Downstream models and heuristics should treat it as one
    weak signal among many.
    """
    clean_url: str = validate_url(url)

    parsed   = urlparse(clean_url)
    extracted = tldextract.extract(clean_url, include_psl_private_domains=False)

    hostname: str      = parsed.hostname or ""
    netloc: str        = parsed.netloc   or ""
    path: str          = parsed.path     or ""
    query: str         = parsed.query    or ""

    subdomain:    str = extracted.subdomain or ""
    domain_label: str = extracted.domain   or ""
    suffix:       str = extracted.suffix   or ""

    port: int | None = parsed.port

    # ------------------------------------------------------------------ #
    # --- Original 12 features (names and semantics preserved exactly) --- #
    # ------------------------------------------------------------------ #

    url_length         = len(clean_url)
    hostname_length    = len(hostname)
    path_length        = len(path)
    query_length       = len(query)
    dot_count          = clean_url.count(".")
    subdomain_depth    = len(subdomain.split(".")) if subdomain else 0
    domain_entropy     = round(_shannon_entropy(domain_label), 6)
    is_ip_host         = int(_is_ip_address(hostname))
    is_https           = int(parsed.scheme.lower() == "https")
    suspicious_kw_count= _count_suspicious_keywords(clean_url.lower())
    has_non_std_port   = int(port is not None and port not in _STANDARD_PORTS)
    has_suspicious_tld = int(suffix.lower() in _SUSPICIOUS_TLDS)

    # ------------------------------------------------------------------ #
    # --- New 14 features                                              --- #
    # ------------------------------------------------------------------ #

    # Total label count in hostname (e.g. a.b.example.com → 4).
    hostname_label_count = len(hostname.split(".")) if hostname else 0

    # Path depth: number of non-empty slash segments.
    path_depth = len([s for s in path.split("/") if s])

    # '@' in authority (user-info impersonation trick).
    has_at_in_authority = int("@" in netloc)

    # Percent-encoded characters in path or query.
    has_encoded_chars = int(
        bool(_PCT_ENCODED_RE.search(path)) or bool(_PCT_ENCODED_RE.search(query))
    )

    # Redirect parameter in query string.
    has_redirect_param = int(_has_redirect_param(query))

    # IDN / punycode label.
    is_idn = int("xn--" in hostname.lower())

    # Minimum Levenshtein distance from registered domain to known brands.
    # A distance of 0 means the domain IS a known brand (could be legit or
    # impersonation depending on TLD/context).  Small non-zero distances
    # (1–3) are the most suspicious.
    brand_min_edit_dist = _brand_min_edit_distance(domain_label)

    # Brand appears in subdomain but NOT the registered domain.
    brand_in_subdomain = int(_brand_in_subdomain(subdomain, domain_label))

    # Digit substitution in the registered domain.
    has_digit_sub_in_domain = int(_has_digit_substitution(domain_label))

    # Brand + hyphen in registered domain.
    has_hyphen_brand = int(_has_hyphen_brand(domain_label))

    # Longest consecutive digit run in registered domain.
    consecutive_digits = _longest_consecutive_digits(domain_label)

    # User-info present in raw netloc.
    has_userinfo = int(_has_userinfo(netloc))

    # Brand name in URL path.
    path_has_brand = int(_path_has_brand(path))

    # Number of labels in the public suffix.
    tld_label_count = _tld_label_count(suffix)

    return {
        # --- original 12 ---
        "url_length":           url_length,
        "hostname_length":      hostname_length,
        "path_length":          path_length,
        "query_length":         query_length,
        "dot_count":            dot_count,
        "subdomain_depth":      subdomain_depth,
        "domain_entropy":       domain_entropy,
        "is_ip_host":           is_ip_host,
        "is_https":             is_https,
        "suspicious_kw_count":  suspicious_kw_count,
        "has_non_std_port":     has_non_std_port,
        "has_suspicious_tld":   has_suspicious_tld,
        # --- new 14 ---
        "hostname_label_count":    hostname_label_count,
        "path_depth":              path_depth,
        "has_at_in_authority":     has_at_in_authority,
        "has_encoded_chars":       has_encoded_chars,
        "has_redirect_param":      has_redirect_param,
        "is_idn":                  is_idn,
        "brand_min_edit_dist":     brand_min_edit_dist,
        "brand_in_subdomain":      brand_in_subdomain,
        "has_digit_sub_in_domain": has_digit_sub_in_domain,
        "has_hyphen_brand":        has_hyphen_brand,
        "consecutive_digits":      consecutive_digits,
        "has_userinfo":            has_userinfo,
        "path_has_brand":          path_has_brand,
        "tld_label_count":         tld_label_count,
    }
