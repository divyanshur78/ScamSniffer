"""
src/features/text_features.py
==============================
Deterministic numeric feature extraction from free-text messages
(email bodies, SMS, and chat messages).

⚠️  This module NEVER opens, fetches, or resolves any link found in the text.
    All analysis is purely lexical.

⚠️  Disclaimer: these heuristic features are lightweight signals that *assist*
    a machine-learning classifier.  They cannot reliably classify phishing on
    their own and must not be used as a standalone decision mechanism.

Public API
----------
extract_text_features(text: str, *, input_type: str = "message") -> dict[str, int | float]
    Validate the text with src.validator and return a fixed-length feature dict.

Feature catalogue
-----------------
text_length         : int   Total character count of the stripped message
word_count          : int   Whitespace-tokenised word count
avg_word_length     : float Mean character length of words (0.0 if no words)
uppercase_ratio     : float Fraction of alpha characters that are uppercase (0.0–1.0)
exclamation_count   : int   Number of '!' characters
question_count      : int   Number of '?' characters
url_count           : int   Number of http/https URL-like patterns in the text
urgency_term_count  : int   Count of distinct urgency/threat terms present
credential_req_count: int   Count of distinct credential-request terms present
money_term_count    : int   Count of distinct money/payment-related terms present
digit_ratio         : float Fraction of characters that are ASCII digits (0.0–1.0)
unique_word_ratio   : float Ratio of unique lowercase words to total words (0.0–1.0)

Assumptions / edge-case handling
---------------------------------
- Input is passed through validate_message(); invalid values raise ValueError.
- Tokenisation is whitespace-only (no NLTK dependency).
- Term matching is whole-word, case-insensitive, using regex word boundaries.
- avg_word_length, uppercase_ratio, digit_ratio, and unique_word_ratio all
  return 0.0 when the denominator is zero (empty-after-strip is already
  rejected by the validator, but guards remain for robustness).
"""

from __future__ import annotations

import re
from src.validator import validate_message

# ---------------------------------------------------------------------------
# Constants — term lists
# ---------------------------------------------------------------------------

# Words that convey urgency or threats commonly used in phishing.
_URGENCY_TERMS: frozenset[str] = frozenset(
    {
        "urgent",
        "immediately",
        "action required",
        "act now",
        "limited time",
        "expires",
        "expiring",
        "deadline",
        "last chance",
        "final notice",
        "suspended",
        "suspend",
        "terminated",
        "termination",
        "locked",
        "restricted",
        "alert",
        "warning",
        "critical",
        "important",
        "respond",
        "asap",
        "right now",
        "do not ignore",
    }
)

# Phrases/words requesting credentials or personal information.
_CREDENTIAL_TERMS: frozenset[str] = frozenset(
    {
        "password",
        "passwd",
        "username",
        "user name",
        "login",
        "log in",
        "sign in",
        "signin",
        "verify",
        "verification",
        "confirm",
        "credential",
        "credentials",
        "social security",
        "ssn",
        "date of birth",
        "bank account",
        "account number",
        "pin",
        "security code",
        "otp",
        "one-time",
        "two-factor",
        "2fa",
    }
)

# Money, payment, and prize-related terms.
_MONEY_TERMS: frozenset[str] = frozenset(
    {
        "payment",
        "invoice",
        "transaction",
        "transfer",
        "wire",
        "bitcoin",
        "crypto",
        "paypal",
        "credit card",
        "debit card",
        "bank",
        "refund",
        "compensation",
        "prize",
        "winner",
        "won",
        "lottery",
        "claim",
        "reward",
        "gift card",
        "voucher",
        "cashback",
        "free",
        "click here",
    }
)

# Regex to find http/https URLs (matches without resolving them).
_URL_PATTERN: re.Pattern[str] = re.compile(
    r"https?://[^\s\"'<>]+",
    re.IGNORECASE,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _count_terms(text_lower: str, terms: frozenset[str]) -> int:
    """Count how many distinct terms from *terms* appear in *text_lower*.

    Each term is matched as a whole phrase using ``\\b`` word boundaries where
    applicable.  Multi-word phrases (containing a space) use a simple substring
    check after boundary anchoring.
    """
    count = 0
    for term in terms:
        # Build a pattern: word boundary on each side, escape special chars.
        escaped = re.escape(term)
        pattern = rf"\b{escaped}\b"
        if re.search(pattern, text_lower):
            count += 1
    return count


def _uppercase_ratio(text: str) -> float:
    """Fraction of alphabetic characters that are uppercase."""
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return 0.0
    return sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)


def _digit_ratio(text: str) -> float:
    """Fraction of total characters that are ASCII digits."""
    if not text:
        return 0.0
    return sum(1 for c in text if c.isdigit()) / len(text)


def _unique_word_ratio(words: list[str]) -> float:
    """Ratio of unique lowercase words to total word count."""
    if not words:
        return 0.0
    return len(set(w.lower() for w in words)) / len(words)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def extract_text_features(
    text: object,
    *,
    input_type: str = "message",
) -> dict[str, int | float]:
    """
    Extract a fixed set of numeric features from a text message.

    Parameters
    ----------
    text:
        Raw input value.  Passed through ``validate_message()`` before
        processing — invalid inputs raise ``ValueError``.
    input_type:
        Human-readable label forwarded to ``validate_message()`` for error
        messages (e.g. ``"email"``, ``"SMS"``, ``"chat message"``).

    Returns
    -------
    dict[str, int | float]
        Feature dictionary with the keys listed in the module docstring.
        All values are numeric; none are ``None``.

    Raises
    ------
    ValueError
        Propagated from ``validate_message()`` for non-string, empty,
        whitespace-only, or oversized inputs.
    """
    clean: str = validate_message(text, input_type=input_type)
    text_lower: str = clean.lower()
    words: list[str] = clean.split()

    avg_word_length: float = (
        sum(len(w) for w in words) / len(words) if words else 0.0
    )

    return {
        "text_length": len(clean),
        "word_count": len(words),
        "avg_word_length": round(avg_word_length, 6),
        "uppercase_ratio": round(_uppercase_ratio(clean), 6),
        "exclamation_count": clean.count("!"),
        "question_count": clean.count("?"),
        "url_count": len(_URL_PATTERN.findall(clean)),
        "urgency_term_count": _count_terms(text_lower, _URGENCY_TERMS),
        "credential_req_count": _count_terms(text_lower, _CREDENTIAL_TERMS),
        "money_term_count": _count_terms(text_lower, _MONEY_TERMS),
        "digit_ratio": round(_digit_ratio(clean), 6),
        "unique_word_ratio": round(_unique_word_ratio(words), 6),
    }
