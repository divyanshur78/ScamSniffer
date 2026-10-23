
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
    _SUSPICIOUS_KEYWORDS,
    extract_url_features,
)
from src.validator import validate_message, validate_url

_DEFAULT_MODEL_DIR = Path("models")
_URL_MODEL_FILE    = "url_model.joblib"
_TEXT_MODEL_FILE   = "text_model.joblib"

_DISCLAIMER = (
    "ScamSniffer predictions are probabilistic and NOT guaranteed to be correct. "
    "Do not rely solely on this tool for security decisions. "
    "Always apply human judgement."
)

_LABEL_TEXT = {0: "legitimate", 1: "phishing"}

_VERDICT_PHISHING  = "Potentially phishing"
_VERDICT_BENIGN    = "No strong signals detected"
_VERDICT_UNABLE    = "Unable to assess"

_HIGH_CONF_HEURISTICS = frozenset({
    "is_ip_host",
    "has_suspicious_tld",
    "has_non_std_port",
    "is_idn",
    "has_userinfo",
    "has_at_in_authority",
})


class InputType(str, enum.Enum):
    URL   = "url"
    EMAIL = "email"
    SMS   = "sms"
    CHAT  = "chat"


class ModelNotReadyError(RuntimeError):
    """Raised when a required model artefact has not been trained/saved yet."""

@dataclass
class DetectionResult:

    input_type    : str
    label         : int
    label_text    : str
    verdict       : str
    confidence    : float
    phishing_prob : float
    warnings      : list[str]
    disclaimer    : str = field(default=_DISCLAIMER)
    severity: str = "Unknown"
    possible_types: list[str] = field(default_factory=list)
    recommended_actions: list[str] = field(default_factory=list)
    combined_risk_score: float = 0.0

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
            "severity":      self.severity,
            "possible_types": self.possible_types,
            "recommended_actions": self.recommended_actions,
            "combined_risk_score": self.combined_risk_score,
        }


def _url_warnings(features: dict[str, int | float], url: str) -> list[str]:

    warnings: list[str] = []
    url_lower = url.lower()

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

    if features.get("is_idn"):
        warnings.append("Hostname uses punycode (IDN) encoding — possible visual spoofing.")
    if features.get("has_at_in_authority") or features.get("has_userinfo"):
        warnings.append(
            "URL contains user-info or '@' in authority — "
            "possible credential injection trick."
        )
    if features.get("has_encoded_chars"):
        warnings.append("URL contains percent-encoded characters in path or query.")
    if features.get("has_redirect_param"):
        warnings.append("URL contains a redirect/forwarding parameter.")

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


def _verdict_for_url(
    label: int,
    features: dict[str, int | float],
) -> str:
    structural_flags = any(features.get(h) for h in _HIGH_CONF_HEURISTICS)

    brand_lookalike = (
        0 < features.get("brand_min_edit_dist", 99) <= 2
        or bool(features.get("has_digit_sub_in_domain"))
        or bool(features.get("brand_in_subdomain"))
        or bool(features.get("has_hyphen_brand"))
        or bool(features.get("path_has_brand"))
    )

    if label == 1 or structural_flags or brand_lookalike:
        return _VERDICT_PHISHING

    return _VERDICT_BENIGN






def _verdict_for_text(
    label: int,
    features: dict[str, int | float],
) -> str:
    urgency = features.get("urgency_term_count", 0) >= 1
    credentials = features.get("credential_req_count", 0) >= 1

    if urgency and credentials:
        return _VERDICT_PHISHING

    if label == 1:
        return _VERDICT_PHISHING

    return _VERDICT_BENIGN




def _assess_risk(
    input_type: str,
    features: dict[str, int | float],
    warnings: list[str],
    label: int,
    phishing_prob: float = 0.0,
) -> tuple[str, list[str], list[str], float]:
    possible_types: list[str] = []
    actions: list[str] = []
    score = 0

    if input_type == "url":
        if features.get("is_ip_host"):
            score += 2
            possible_types.append("IP-address-based URL")
        if features.get("has_suspicious_tld"):
            score += 1
            possible_types.append("Suspicious domain ending")
        if features.get("is_idn") or features.get("has_encoded_chars"):
            score += 2
            possible_types.append("Possible URL obfuscation")
        if features.get("has_userinfo") or features.get("has_at_in_authority"):
            score += 2
            possible_types.append("Possible URL authority deception")
        if features.get("has_redirect_param"):
            score += 1
            possible_types.append("Possible redirect or forwarding")
        if features.get("brand_in_subdomain") or features.get("has_hyphen_brand"):
            score += 2
            possible_types.append("Possible brand impersonation")
        if features.get("has_digit_sub_in_domain"):
            score += 2
            possible_types.append("Possible lookalike or typosquatting")
        if 0 < features.get("brand_min_edit_dist", 99) <= 2:
            score += 1
            if "Possible lookalike or typosquatting" not in possible_types:
                possible_types.append("Possible lookalike or typosquatting")
        if features.get("suspicious_kw_count", 0) >= 2:
            score += 1
            possible_types.append("Credential or account-themed URL")
        if not features.get("is_https"):
            score += 1
            possible_types.append("Unencrypted HTTP connection")
    else:
        if features.get("credential_req_count", 0) >= 1:
            score += 2
            possible_types.append("Possible credential harvesting")
        if features.get("urgency_term_count", 0) >= 1:
            score += 1
            possible_types.append("Urgency or account-threat pressure")
        if features.get("money_term_count", 0) >= 1:
            score += 1
            possible_types.append("Possible prize or financial scam")
        if features.get("url_count", 0) >= 1:
            score += 1
            possible_types.append("Message contains embedded link")

    heuristic_score = min(score / 7.0, 1.0)
    model_score = max(0.0, min(float(phishing_prob), 1.0))
    combined_score = (0.60 * model_score) + (0.40 * heuristic_score)

    lookalike = (
        bool(features.get("has_digit_sub_in_domain"))
        or bool(features.get("brand_in_subdomain"))
        or bool(features.get("has_hyphen_brand"))
        or 0 < features.get("brand_min_edit_dist", 99) <= 2
    )
    if input_type == "url" and lookalike:
        combined_score = max(combined_score, 0.35)

    combined_risk_score = round(combined_score * 100, 1)

    if combined_risk_score >= 80:
        severity = "Critical"
    elif combined_risk_score >= 50:
        severity = "High"
    elif combined_risk_score >= 30:
        severity = "Medium"
    elif combined_risk_score >= 10:
        severity = "Low"
    else:
        severity = "Low"

    if label == 1 and not possible_types:
        possible_types.append("Model detected suspicious patterns")
    if not possible_types and warnings:
        possible_types.append("Other suspicious indicators")
    if not possible_types:
        possible_types.append("No specific phishing pattern identified")

    if severity in {"Critical", "High"}:
        actions.append(
            "Do not open the link or provide credentials or payment details."
        )
    elif severity == "Medium":
        actions.append(
            "Verify the sender and destination domain independently before acting."
        )
    else:
        actions.append(
            "No strong indicators were found, but this does not guarantee safety."
        )

    actions.append(
        "If the message claims to be from an organisation, use its official app or website."
    )

    return severity, possible_types, actions, combined_risk_score


class Detector:

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

    def detect(self, input_value: object, input_type: str | InputType) -> DetectionResult:

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

    def _detect_url(self, raw: object) -> DetectionResult:
        """Validate, extract features, run URL model, build result."""
        clean    = validate_url(raw)
        features = extract_url_features(clean)

        proba         = self._url_model.predict_proba([features])[0]
        phishing_prob = float(proba[1])
        label         = int(self._url_model.predict([features])[0])
        confidence    = float(proba[label])
        url_warnings  = _url_warnings(features, clean)
        verdict       = _verdict_for_url(label, features)
        severity, possible_types, recommended_actions, combined_risk_score = (
    _assess_risk("url", features, url_warnings, label, phishing_prob)
)

        return DetectionResult(
            input_type    = InputType.URL.value,
            label         = label,
            label_text    = _LABEL_TEXT[label],
            verdict       = verdict,
            confidence    = confidence,
            phishing_prob = phishing_prob,
            warnings      = url_warnings,
            severity=severity,
            possible_types=possible_types,
            recommended_actions=recommended_actions,
            combined_risk_score=combined_risk_score,
        )

    def _detect_text(self, raw: object, itype: InputType) -> DetectionResult:
        clean    = validate_message(raw, input_type=itype.value)  # raises ValueError if bad
        features = extract_text_features(clean, input_type=itype.value)

        proba         = self._text_model.predict_proba([clean])[0]
        phishing_prob = float(proba[1])
        label         = int(self._text_model.predict([clean])[0])
        confidence    = float(proba[label])
        txt_warnings  = _text_warnings(features, clean)
        verdict       = _verdict_for_text(label, features)
        severity, possible_types, recommended_actions, combined_risk_score = (
    _assess_risk("text", features, txt_warnings, label, phishing_prob)
)

        return DetectionResult(
            input_type    = itype.value,
            label         = label,
            label_text    = _LABEL_TEXT[label],
            verdict       = verdict,
            confidence    = confidence,
            phishing_prob = phishing_prob,
            warnings      = txt_warnings,
            severity=severity,
            possible_types=possible_types,
            recommended_actions=recommended_actions,
            combined_risk_score=combined_risk_score,
        )


def detect_url(url: object, model_dir: Path | str = _DEFAULT_MODEL_DIR) -> DetectionResult:

    return Detector(model_dir).detect(url, InputType.URL)


def detect_message(
    text: object,
    input_type: str | InputType = InputType.EMAIL,
    model_dir: Path | str = _DEFAULT_MODEL_DIR,
) -> DetectionResult:

    return Detector(model_dir).detect(text, input_type)
