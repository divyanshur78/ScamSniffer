"""
src/validator.py
================
Input validation for ScamSniffer.

All public functions accept a raw value, apply normalisation, and return the
cleaned string — or raise ValueError with a descriptive message.

Rules enforced for every input type:
  - Must be a str (not None, int, list, …)
  - Must not be empty or whitespace-only after stripping
  - Leading/trailing whitespace is stripped before the value is returned
  - Maximum length limits are checked *after* stripping

URL-specific additional rules:
  - Maximum 2 083 characters (Internet Explorer legacy limit, widely adopted)
  - Must start with http:// or https:// after normalisation

Message (email / SMS / chat) additional rules:
  - Maximum 10 000 characters

⚠️  This module NEVER fetches, executes, or resolves any submitted URL.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

URL_MAX_LENGTH: int = 2_083
MESSAGE_MAX_LENGTH: int = 10_000

_ALLOWED_URL_SCHEMES: tuple[str, ...] = ("http://", "https://")


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _assert_non_empty_string(value: object, label: str) -> str:
    """
    Verify *value* is a non-empty, non-whitespace string.

    Returns the stripped string on success.
    Raises ValueError with a clear message on failure.
    """
    if value is None:
        raise ValueError(f"{label} must be a string, got None.")
    if not isinstance(value, str):
        raise ValueError(
            f"{label} must be a string, got {type(value).__name__!r}."
        )
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"{label} must not be empty or whitespace-only.")
    return stripped


# ---------------------------------------------------------------------------
# Public validators
# ---------------------------------------------------------------------------


def validate_url(value: object) -> str:
    """
    Validate and normalise a URL string.

    Parameters
    ----------
    value:
        Raw input from the caller.

    Returns
    -------
    str
        Stripped URL ready for feature extraction.

    Raises
    ------
    ValueError
        If *value* is not a string, is empty/whitespace, exceeds
        ``URL_MAX_LENGTH``, or does not start with http:// or https://.
    """
    url = _assert_non_empty_string(value, "URL")

    if len(url) > URL_MAX_LENGTH:
        raise ValueError(
            f"URL exceeds the maximum allowed length of {URL_MAX_LENGTH} characters "
            f"(got {len(url)})."
        )

    lower = url.lower()
    if not any(lower.startswith(scheme) for scheme in _ALLOWED_URL_SCHEMES):
        raise ValueError(
            "URL must start with 'http://' or 'https://'."
        )

    return url


def validate_message(value: object, *, input_type: str = "message") -> str:
    """
    Validate and normalise a free-text message (email body, SMS, or chat).

    Parameters
    ----------
    value:
        Raw input from the caller.
    input_type:
        Human-readable label used in error messages (e.g. ``"email"``,
        ``"SMS"``, ``"chat message"``).  Defaults to ``"message"``.

    Returns
    -------
    str
        Stripped message text ready for classification.

    Raises
    ------
    ValueError
        If *value* is not a string, is empty/whitespace, or exceeds
        ``MESSAGE_MAX_LENGTH``.
    """
    label = input_type.capitalize()
    text = _assert_non_empty_string(value, label)

    if len(text) > MESSAGE_MAX_LENGTH:
        raise ValueError(
            f"{label} exceeds the maximum allowed length of {MESSAGE_MAX_LENGTH} "
            f"characters (got {len(text)})."
        )

    return text
