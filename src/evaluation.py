# src/evaluation.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

import matplotlib.pyplot as plt


@dataclass
class EvalResult:
    metrics: Dict[str, float]
    confusion: np.ndarray
    report: Dict[str, Any]


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
) -> EvalResult:
    metrics: Dict[str, float] = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }

    if y_prob is not None:
        # May error if only one class present; guard.
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except Exception:
            metrics["roc_auc"] = float("nan")
        try:
            metrics["avg_precision"] = float(average_precision_score(y_true, y_prob))
        except Exception:
            metrics["avg_precision"] = float("nan")

    cm = confusion_matrix(y_true, y_pred)
    rep = classification_report(y_true, y_pred, output_dict=True, zero_division=0)

    return EvalResult(metrics=metrics, confusion=cm, report=rep)


def plot_roc(y_true: np.ndarray, y_prob: np.ndarray, *, title: str = "ROC Curve") -> plt.Figure:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig = plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(title)
    return fig


def plot_precision_recall(y_true: np.ndarray, y_prob: np.ndarray, *, title: str = "Precision-Recall Curve") -> plt.Figure:
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    fig = plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(title)
    return fig


def find_best_threshold_f1(y_true: np.ndarray, y_prob: np.ndarray) -> Tuple[float, float]:
    """
    Returns (best_threshold, best_f1).
    """
    thresholds = np.linspace(0.05, 0.95, 19)
    best_thr = 0.5
    best_f1 = -1.0
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    return best_thr, float(best_f1)
