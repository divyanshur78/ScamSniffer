
from __future__ import annotations

import re

from src.validator import validate_message

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

_URL_PATTERN: re.Pattern[str] = re.compile(
    r"https?://[^\s\"'<>]+",
    re.IGNORECASE,
)



def _count_terms(text_lower: str, terms: frozenset[str]) -> int:

    count = 0
    for term in terms:
        escaped = re.escape(term)
        pattern = rf"\b{escaped}\b"
        if re.search(pattern, text_lower):
            count += 1
    return count

def _uppercase_ratio(text: str) -> float:
    alpha_chars = [c for c in text if c.isalpha()]
    if not alpha_chars:
        return 0.0
    return sum(1 for c in alpha_chars if c.isupper()) / len(alpha_chars)

def _digit_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if c.isdigit()) / len(text)

def _unique_word_ratio(words: list[str]) -> float:
    if not words:
        return 0.0
    return len(set(w.lower() for w in words)) / len(words)

def extract_text_features(
    text: object,
    *,
    input_type: str = "message",
) -> dict[str, int | float]:

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
