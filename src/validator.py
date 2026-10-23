
from __future__ import annotations

URL_MAX_LENGTH: int = 2_083
MESSAGE_MAX_LENGTH: int = 10_000

_ALLOWED_URL_SCHEMES: tuple[str, ...] = ("http://", "https://")


def _assert_non_empty_string(value: object, label: str) -> str:

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


def validate_url(value: object) -> str:

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

    label = input_type.capitalize()
    text = _assert_non_empty_string(value, label)

    if len(text) > MESSAGE_MAX_LENGTH:
        raise ValueError(
            f"{label} exceeds the maximum allowed length of {MESSAGE_MAX_LENGTH} "
            f"characters (got {len(text)})."
        )

    return text
