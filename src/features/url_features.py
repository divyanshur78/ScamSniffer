
from __future__ import annotations

import math
import re
from collections import Counter
from urllib.parse import urlparse

import tldextract

from src.validator import validate_url

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

_STANDARD_PORTS: frozenset[int] = frozenset({80, 443})

_SUSPICIOUS_TLDS: frozenset[str] = frozenset(
    {
        "tk",
        "ml",
        "ga",
        "cf",
        "gq",
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

_DIGIT_SUB_MAP: dict[str, str] = {
    "0": "o",
    "1": "li",
    "3": "e",
    "4": "a",
    "5": "s",
    "6": "g",
    "8": "b",
}

_IPV4_RE: re.Pattern[str] = re.compile(
    r"^(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)$"
)

_PCT_ENCODED_RE: re.Pattern[str] = re.compile(r"%[0-9A-Fa-f]{2}")

_USERINFO_RE: re.Pattern[str] = re.compile(r"^[^@/]+@")

_CONSECUTIVE_DIGITS_RE: re.Pattern[str] = re.compile(r"\d+")

def _shannon_entropy(text: str) -> float:

    if not text:
        return 0.0
    counts = Counter(text)
    length = len(text)
    return -sum(
        (c / length) * math.log2(c / length) for c in counts.values()
    )


def _is_ip_address(hostname: str) -> bool:

    if not hostname:
        return False
    if ":" in hostname:
        return True
    return bool(_IPV4_RE.match(hostname))


def _count_suspicious_keywords(url_lower: str) -> int:
    return sum(1 for kw in _SUSPICIOUS_KEYWORDS if kw in url_lower)


def _levenshtein(s: str, t: str) -> int:

    if s == t:
        return 0
    if not s:
        return len(t)
    if not t:
        return len(s)
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

    if not domain_label:
        return 99.0
    dl = domain_label.lower()
    return float(min(_levenshtein(dl, brand) for brand in _KNOWN_BRANDS))


def _brand_in_subdomain(subdomain: str, domain_label: str) -> bool:

    if not subdomain:
        return False
    sub_lower    = subdomain.lower()
    domain_lower = domain_label.lower()
    for brand in _KNOWN_BRANDS:
        if brand in sub_lower and brand not in domain_lower:
            return True
    return False


def _has_digit_substitution(domain_label: str) -> bool:

    if not domain_label:
        return False
    dl = domain_label.lower()

    sub_digits = [c for c in dl if c in _DIGIT_SUB_MAP]
    if not sub_digits:
        return False

    substituted = dl
    for digit, letters in _DIGIT_SUB_MAP.items():
        substituted = substituted.replace(digit, letters[0])

    min_dist = _brand_min_edit_distance(substituted)
    return min_dist <= 2


def _has_hyphen_brand(domain_label: str) -> bool:

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

    if not domain_label:
        return 0
    matches = _CONSECUTIVE_DIGITS_RE.findall(domain_label)
    return max((len(m) for m in matches), default=0)


def _has_redirect_param(query: str) -> bool:

    if not query:
        return False
    q_lower = query.lower()

    for pair in re.split(r"[&;]", q_lower):
        key = pair.split("=", 1)[0].strip()
        if key in _REDIRECT_PARAMS:
            return True
    return False


def _has_userinfo(netloc: str) -> bool:

    return bool(_USERINFO_RE.match(netloc))


def _path_has_brand(path: str) -> bool:

    if not path:
        return False
    path_lower = path.lower()
    for brand in _KNOWN_BRANDS:
        if brand in path_lower:
            return True
    return False


def _tld_label_count(suffix: str) -> int:

    if not suffix:
        return 0
    return len(suffix.split("."))



def extract_url_features(url: object) -> dict[str, int | float]:

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

    hostname_label_count = len(hostname.split(".")) if hostname else 0

    path_depth = len([s for s in path.split("/") if s])

    has_at_in_authority = int("@" in netloc)

    has_encoded_chars = int(
        bool(_PCT_ENCODED_RE.search(path)) or bool(_PCT_ENCODED_RE.search(query))
    )

    has_redirect_param = int(_has_redirect_param(query))

    is_idn = int("xn--" in hostname.lower())

    brand_min_edit_dist = _brand_min_edit_distance(domain_label)

    brand_in_subdomain = int(_brand_in_subdomain(subdomain, domain_label))

    has_digit_sub_in_domain = int(_has_digit_substitution(domain_label))

    has_hyphen_brand = int(_has_hyphen_brand(domain_label))

    consecutive_digits = _longest_consecutive_digits(domain_label)

    has_userinfo = int(_has_userinfo(netloc))

    path_has_brand = int(_path_has_brand(path))

    tld_label_count = _tld_label_count(suffix)

    return {
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
