"""
src/evaluation.py
=================
Model evaluation utilities for ScamSniffer.

Provides a single public function:

    evaluate(y_true, y_pred, y_proba=None) -> EvaluationResult

which returns a dataclass containing precision, recall, F1-score,
confusion matrix, class counts, and optionally ROC-AUC.

⚠️  Metrics computed here reflect *only the dataset passed in*.
    Synthetic-data results are not representative of real-world performance.
    Do not use scores from this module as claims of production accuracy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------


@dataclass
class EvaluationResult:
    """Container for evaluation metrics returned by :func:`evaluate`."""

    precision: float
    recall: float
    f1: float
    confusion_matrix: list[list[int]]
    class_counts: dict[str, int]
    classification_report: str
    roc_auc: float | None = field(default=None)

    def as_dict(self) -> dict:
        """Return a JSON-serialisable representation."""
        return {
            "precision": round(self.precision, 6),
            "recall": round(self.recall, 6),
            "f1": round(self.f1, 6),
            "roc_auc": round(self.roc_auc, 6) if self.roc_auc is not None else None,
            "confusion_matrix": self.confusion_matrix,
            "class_counts": self.class_counts,
            "classification_report": self.classification_report,
        }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def evaluate(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_proba: Sequence[float] | None = None,
) -> EvaluationResult:
    """
    Compute evaluation metrics from true and predicted labels.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels (0 = legitimate, 1 = phishing).
    y_pred:
        Predicted binary labels.
    y_proba:
        Optional predicted probability of the positive class (label 1).
        Required for ROC-AUC; ignored when fewer than 2 classes are present
        in y_true (avoids undefined-metric errors on trivial test splits).

    Returns
    -------
    EvaluationResult
        Dataclass with precision, recall, f1, confusion_matrix,
        class_counts, classification_report, and roc_auc (or None).

    Raises
    ------
    ValueError
        If y_true or y_pred are empty, have different lengths, or contain
        values outside {0, 1}.
    """
    y_true_arr = np.asarray(y_true, dtype=int)
    y_pred_arr = np.asarray(y_pred, dtype=int)

    # --- input guards ---
    if len(y_true_arr) == 0:
        raise ValueError("y_true must not be empty.")
    if len(y_true_arr) != len(y_pred_arr):
        raise ValueError(
            f"y_true (len={len(y_true_arr)}) and y_pred (len={len(y_pred_arr)}) "
            "must have the same length."
        )
    for name, arr in (("y_true", y_true_arr), ("y_pred", y_pred_arr)):
        bad = set(arr.tolist()) - {0, 1}
        if bad:
            raise ValueError(
                f"{name} contains values outside {{0, 1}}: {sorted(bad)}."
            )

    unique_classes = sorted(set(y_true_arr.tolist()))
    n_classes = len(unique_classes)

    # zero_division=0 avoids warnings on tiny splits where a class is absent.
    precision = float(
        precision_score(y_true_arr, y_pred_arr, zero_division=0)
    )
    recall = float(
        recall_score(y_true_arr, y_pred_arr, zero_division=0)
    )
    f1 = float(
        f1_score(y_true_arr, y_pred_arr, zero_division=0)
    )
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1]).tolist()
    report = classification_report(
        y_true_arr, y_pred_arr,
        labels=[0, 1],
        target_names=["legitimate", "phishing"],
        zero_division=0,
    )
    class_counts = {
        "legitimate (0)": int((y_true_arr == 0).sum()),
        "phishing (1)": int((y_true_arr == 1).sum()),
    }

    # ROC-AUC only defined when both classes present and probabilities supplied.
    roc_auc: float | None = None
    if y_proba is not None and n_classes == 2:
        roc_auc = float(roc_auc_score(y_true_arr, np.asarray(y_proba)))

    return EvaluationResult(
        precision=precision,
        recall=recall,
        f1=f1,
        confusion_matrix=cm,
        class_counts=class_counts,
        classification_report=report,
        roc_auc=roc_auc,
    )
