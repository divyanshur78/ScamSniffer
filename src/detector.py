"""
src/detector.py
===============
Unified detection facade for ScamSniffer.

Public API
----------
Detector(model_dir)
    Load saved URL and text models from *model_dir* and expose a single
    ``detect(input_value, input_type)`` method.

detect_url(url, model_dir)     — convenience one-shot function
detect_message(text, input_type, model_dir) — convenience one-shot function

InputType
    Enum of accepted input types: URL, EMAIL, SMS, CHAT.

DetectionResult
    Dataclass returned by every detection call.  Fields:

        input_type  : str        — one of "url", "email", "sms", "chat"
        label       : int        — 0 = legitimate, 1 = phishing/scam
        label_text  : str        — "legitimate" or "phishing"
        verdict     : str        — three-tier: "Potentially phishing",
                                   "No strong signals detected", or
                                   "Unable to assess"
        confidence  : float      — probability of the predicted class (0–1)
        phishing_prob : float    — P(phishing) from the model (0–1)
        warnings    : list[str]  — human-readable indicators that raised flags
        disclaimer  : str        — always-present safety disclaimer

Three-tier verdict
------------------
The verdict is determined independently from the model label:

  "Potentially phishing"      — model predicts phishing (label=1), OR one or
                                 more high-confidence heuristic warnings fired.
  "No strong signals detected" — model predicts legitimate AND no
                                 high-confidence heuristic warnings fired.
  "Unable to assess"          — feature extraction failed or model unavailable.

⚠️  "No strong signals detected" is NOT a guarantee of safety.  The model
    operates on a limited synthetic training set.  Always apply human judgement.

Design constraints
------------------
- NEVER fetches, executes, or resolves any submitted URL.
- Input is validated by src.validator before any processing.
- Model probability is NOT a guarantee of safety.  The disclaimer field
  repeats this on every result.
- A missing model artefact raises ModelNotReadyError with a clear setup message.
- Heuristic warnings are generated independently of the model score.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import joblib

from src.features.text_features import (
    _CREDENTIAL_TERMS,
    _MONEY_TERMS,
    _URGENCY_TERMS,
    extract_text_features,
)
from src.features.url_features import (
    _KNOWN_BRANDS,
    _SUSPICIOUS_KEYWORDS,
    _SUSPICIOUS_TLDS,
    extract_url_features,
)
from src.validator import validate_message, validate_url

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_MODEL_DIR = Path("models")
_URL_MODEL_FILE    = "url_model.joblib"
_TEXT_MODEL_FILE   = "text_model.joblib"

_DISCLAIMER = (
    "ScamSniffer predictions are probabilistic and NOT guaranteed to be correct. "
    "Do not rely solely on this tool for security decisions. "
    "Always apply human judgement."
)

_LABEL_TEXT = {0: "legitimate", 1: "phishing"}

# Verdict strings — three-tier system.
_VERDICT_PHISHING  = "Potentially phishing"
_VERDICT_BENIGN    = "No strong signals detected"
_VERDICT_UNABLE    = "Unable to assess"

# Heuristic thresholds — used by _verdict_from_result(), independent of
# model score.  These are WARNING-SIGNAL thresholds; they do NOT directly
# classify the URL as phishing.
_HIGH_CONF_HEURISTICS = frozenset({
    "is_ip_host",
    "has_suspicious_tld",
    "has_non_std_port",
    "is_idn",
    "has_userinfo",
    "has_at_in_authority",
})


# ---------------------------------------------------------------------------
# Enums and exceptions
# ---------------------------------------------------------------------------


class InputType(str, enum.Enum):
    URL   = "url"
    EMAIL = "email"
    SMS   = "sms"
    CHAT  = "chat"


class ModelNotReadyError(RuntimeError):
    """Raised when a required model artefact has not been trained/saved yet."""


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class DetectionResult:
    """
    Result returned by every detection call.

    Attributes
    ----------
    input_type    : str    One of "url", "email", "sms", "chat".
    label         : int    Predicted class — 0 (legitimate) or 1 (phishing).
    label_text    : str    Human-readable label — "legitimate" or "phishing".
    verdict       : str    Three-tier verdict string (see module docstring).
    confidence    : float  Probability of the *predicted* class (0.0–1.0).
    phishing_prob : float  P(phishing) from the model regardless of prediction.
    warnings      : list   Human-readable indicators that contributed to risk.
    disclaimer    : str    Always-present safety caveat.
    """

    input_type    : str
    label         : int
    label_text    : str
    verdict       : str
    confidence    : float
    phishing_prob : float
    warnings      : list[str]
    disclaimer    : str = field(default=_DISCLAIMER)

    def as_dict(self) -> dict[str, Any]:
        return {
            "input_type":    self.input_type,
            "label":         self.label,
            "label_text":    self.label_text,
            "verdict":       self.verdict,
            "confidence":    round(self.confidence, 4),
            "phishing_prob": round(self.phishing_prob, 4),
            "warnings":      self.warnings,
            "disclaimer":    self.disclaimer,
        }


# ---------------------------------------------------------------------------
# Warning generators (purely lexical — no network access)
# ---------------------------------------------------------------------------


def _url_warnings(features: dict[str, int | float], url: str) -> list[str]:
    """
    Return a list of human-readable warning strings for a URL.

    ⚠️  These warnings are generated INDEPENDENTLY of the model score.
        The presence of a warning does not directly classify the URL as
        phishing.  Brand-similarity warnings are signals, not proof.
    """
    warnings: list[str] = []
    url_lower = url.lower()

    # --- Structural / protocol warnings ---
    if features.get("is_ip_host"):
        warnings.append("Hostname is an IP address (not a domain name).")
    if not features.get("is_https"):
        warnings.append("URL uses HTTP, not HTTPS.")
    if features.get("has_suspicious_tld"):
        warnings.append("Top-level domain is associated with high phishing rates.")
    if features.get("has_non_std_port"):
        warnings.append("Non-standard port detected in URL.")
    if features.get("subdomain_depth", 0) >= 3:
        warnings.append("Unusually deep subdomain structure.")
    if features.get("domain_entropy", 0) > 3.5:
        warnings.append("Domain name has high character entropy (possible random generation).")
    kw_count = features.get("suspicious_kw_count", 0)
    if kw_count >= 2:
        matched = [kw for kw in _SUSPICIOUS_KEYWORDS if kw in url_lower]
        warnings.append(
            f"URL contains {kw_count} suspicious keyword(s): "
            + ", ".join(sorted(matched)[:5])
            + ("…" if len(matched) > 5 else "")
            + "."
        )
    if features.get("url_length", 0) > 100:
        warnings.append("URL is unusually long.")

    # --- Obfuscation / deception warnings (Phase 2 features) ---
    if features.get("is_idn"):
        warnings.append("Hostname uses punycode (IDN) encoding — possible visual spoofing.")
    if features.get("has_at_in_authority") or features.get("has_userinfo"):
        warnings.append("URL contains user-info or '@' in authority — possible credential injection trick.")
    if features.get("has_encoded_chars"):
        warnings.append("URL contains percent-encoded characters in path or query.")
    if features.get("has_redirect_param"):
        warnings.append("URL contains a redirect/forwarding parameter.")

    # --- Brand-similarity warnings (signal only, NOT proof of phishing) ---
    edit_dist = features.get("brand_min_edit_dist", 99)
    if isinstance(edit_dist, (int, float)) and 0 < edit_dist <= 2:
        warnings.append(
            f"Domain name is very similar to a known brand (edit distance {int(edit_dist)}). "
            "This may indicate a lookalike/typosquatting domain."
        )
    if features.get("has_digit_sub_in_domain"):
        warnings.append(
            "Domain contains digit substitutions (e.g. '0'→'o', '1'→'l') near a brand name — "
            "possible typosquatting."
        )
    if features.get("brand_in_subdomain"):
        warnings.append(
            "A well-known brand name appears in the subdomain but not the registered domain — "
            "possible impersonation (e.g. amazon.attacker.com)."
        )
    if features.get("has_hyphen_brand"):
        warnings.append(
            "Registered domain contains a brand name joined by a hyphen — "
            "common in phishing domains (e.g. paypal-secure)."
        )
    if features.get("path_has_brand") and not features.get("brand_min_edit_dist", 99) == 0:
        warnings.append(
            "A brand name appears in the URL path — the domain itself may not be that brand."
        )
    if features.get("consecutive_digits", 0) >= 4:
        warnings.append(
            f"Domain contains {int(features['consecutive_digits'])} consecutive digits — "
            "possible DGA (domain generation algorithm) pattern."
        )

    return warnings


def _text_warnings(features: dict[str, int | float], text: str) -> list[str]:
    """Return a list of human-readable warning strings for a text message."""
    warnings: list[str] = []
    text_lower = text.lower()

    if features.get("urgency_term_count", 0) >= 1:
        matched = [t for t in _URGENCY_TERMS if t in text_lower]
        warnings.append(
            "Message contains urgency/threat language: "
            + ", ".join(sorted(matched)[:4])
            + ("…" if len(matched) > 4 else "")
            + "."
        )
    if features.get("credential_req_count", 0) >= 1:
        matched = [t for t in _CREDENTIAL_TERMS if t in text_lower]
        warnings.append(
            "Message requests credentials or personal information: "
            + ", ".join(sorted(matched)[:4])
            + ("…" if len(matched) > 4 else "")
            + "."
        )
    if features.get("money_term_count", 0) >= 1:
        matched = [t for t in _MONEY_TERMS if t in text_lower]
        warnings.append(
            "Message contains financial/prize language: "
            + ", ".join(sorted(matched)[:4])
            + ("…" if len(matched) > 4 else "")
            + "."
        )
    if features.get("url_count", 0) >= 1:
        warnings.append(
            f"Message contains {int(features['url_count'])} embedded URL(s)."
        )
    if features.get("uppercase_ratio", 0.0) > 0.4:
        warnings.append("Message contains a high proportion of uppercase characters.")
    if features.get("exclamation_count", 0) >= 3:
        warnings.append(
            f"Message contains {int(features['exclamation_count'])} exclamation marks."
        )
    return warnings


# ---------------------------------------------------------------------------
# Verdict logic (independent of model label)
# ---------------------------------------------------------------------------


def _verdict_for_url(label: int, features: dict[str, int | float]) -> str:
    """
    Compute three-tier verdict for a URL result.

    The verdict is "Potentially phishing" if:
      - The model predicts phishing (label == 1), OR
      - One or more high-confidence structural heuristics fired.

    It is "No strong signals detected" only when the model predicts
    legitimate AND no structural heuristic fired.

    ⚠️  "No strong signals detected" is NOT a safety guarantee.

    Note on brand_min_edit_dist == 0:
        An edit distance of 0 means the registered domain IS a known brand
        (e.g. amazon.com → "amazon").  This is expected for legitimate brand
        domains and must NOT trigger the phishing verdict on its own.
        Only structural heuristics (IP, IDN, non-std port, etc.) can do that
        independently of the label.
    """
    heuristic_hit = any(features.get(h) for h in _HIGH_CONF_HEURISTICS)

    if label == 1 or heuristic_hit:
        return _VERDICT_PHISHING
    return _VERDICT_BENIGN


def _verdict_for_text(label: int, features: dict[str, int | float]) -> str:
    """
    Compute three-tier verdict for a text result.

    "Potentially phishing" if model predicts phishing OR multiple high-signal
    heuristics fired (urgency + credential request together).
    """
    urgency   = features.get("urgency_term_count", 0) >= 1
    cred_req  = features.get("credential_req_count", 0) >= 1

    if label == 1 or (urgency and cred_req):
        return _VERDICT_PHISHING
    return _VERDICT_BENIGN


# ---------------------------------------------------------------------------
# Detector class
# ---------------------------------------------------------------------------


class Detector:
    """
    Unified phishing/scam detector.

    Parameters
    ----------
    model_dir:
        Path to the directory containing ``url_model.joblib`` and
        ``text_model.joblib``.  Defaults to ``models/``.

    Raises
    ------
    ModelNotReadyError
        If either model file is absent.  Run ``scripts/train.py`` first.
    """

    def __init__(self, model_dir: Path | str = _DEFAULT_MODEL_DIR) -> None:
        self._model_dir = Path(model_dir)
        self._url_model  = self._load_model(_URL_MODEL_FILE)
        self._text_model = self._load_model(_TEXT_MODEL_FILE)

    def _load_model(self, filename: str) -> Any:
        path = self._model_dir / filename
        if not path.exists():
            raise ModelNotReadyError(
                f"Model artefact not found: {path}. "
                "Run 'python scripts/train.py' to train and save models first."
            )
        return joblib.load(path)

    # ------------------------------------------------------------------
    # Public detection interface
    # ------------------------------------------------------------------

    def detect(self, input_value: object, input_type: str | InputType) -> DetectionResult:
        """
        Analyse *input_value* and return a :class:`DetectionResult`.

        Parameters
        ----------
        input_value:
            The URL, email body, SMS, or chat message to analyse.
        input_type:
            One of ``"url"``, ``"email"``, ``"sms"``, ``"chat"``
            (case-insensitive), or an :class:`InputType` enum member.

        Returns
        -------
        DetectionResult

        Raises
        ------
        ValueError
            If *input_type* is unrecognised or *input_value* fails validation.
        """
        # Normalise input_type.  If an InputType enum member is passed directly,
        # use its .value; otherwise coerce the raw string to lower-case.
        if isinstance(input_type, InputType):
            itype = input_type
        else:
            try:
                itype = InputType(str(input_type).lower())
            except ValueError:
                raise ValueError(
                    f"Unknown input_type {input_type!r}. "
                    f"Must be one of: {[m.value for m in InputType]}."
                )

        if itype is InputType.URL:
            return self._detect_url(input_value)
        return self._detect_text(input_value, itype)

    # ------------------------------------------------------------------
    # Internal routing
    # ------------------------------------------------------------------

    def _detect_url(self, raw: object) -> DetectionResult:
        """Validate, extract features, run URL model, build result."""
        clean    = validate_url(raw)                      # raises ValueError if bad
        features = extract_url_features(clean)

        proba         = self._url_model.predict_proba([features])[0]  # shape (2,)
        phishing_prob = float(proba[1])
        label         = int(self._url_model.predict([features])[0])
        confidence    = float(proba[label])
        url_warnings  = _url_warnings(features, clean)
        verdict       = _verdict_for_url(label, features)

        return DetectionResult(
            input_type    = InputType.URL.value,
            label         = label,
            label_text    = _LABEL_TEXT[label],
            verdict       = verdict,
            confidence    = confidence,
            phishing_prob = phishing_prob,
            warnings      = url_warnings,
        )

    def _detect_text(self, raw: object, itype: InputType) -> DetectionResult:
        """Validate, run text model, build result."""
        clean    = validate_message(raw, input_type=itype.value)  # raises ValueError if bad
        features = extract_text_features(clean, input_type=itype.value)

        proba         = self._text_model.predict_proba([clean])[0]
        phishing_prob = float(proba[1])
        label         = int(self._text_model.predict([clean])[0])
        confidence    = float(proba[label])
        txt_warnings  = _text_warnings(features, clean)
        verdict       = _verdict_for_text(label, features)

        return DetectionResult(
            input_type    = itype.value,
            label         = label,
            label_text    = _LABEL_TEXT[label],
            verdict       = verdict,
            confidence    = confidence,
            phishing_prob = phishing_prob,
            warnings      = txt_warnings,
        )


# ---------------------------------------------------------------------------
# Convenience one-shot functions
# ---------------------------------------------------------------------------


def detect_url(url: object, model_dir: Path | str = _DEFAULT_MODEL_DIR) -> DetectionResult:
    """
    Convenience wrapper: build a :class:`Detector` and analyse *url*.

    ⚠️  For repeated calls, instantiate :class:`Detector` once and reuse it
    to avoid loading models from disk on every call.
    """
    return Detector(model_dir).detect(url, InputType.URL)


def detect_message(
    text: object,
    input_type: str | InputType = InputType.EMAIL,
    model_dir: Path | str = _DEFAULT_MODEL_DIR,
) -> DetectionResult:
    """
    Convenience wrapper: build a :class:`Detector` and analyse *text*.

    ⚠️  For repeated calls, instantiate :class:`Detector` once and reuse it.
    """
    return Detector(model_dir).detect(text, input_type)
