# src/models.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.svm import LinearSVC

from .config import ModelConfig


def _maybe_calibrate(model, X: csr_matrix, y: np.ndarray, cfg: ModelConfig):
    """
    Calibrate models that do not expose reliable probabilities (e.g., LinearSVC),
    or calibrate any model for better decision quality.
    """
    if not cfg.calibrate:
        return model

    calibrated = CalibratedClassifierCV(
        estimator=model,
        method=cfg.calibration_method,
        cv=cfg.calibration_cv,
    )
    calibrated.fit(X, y)
    return calibrated


def train_logistic_regression(
    X: csr_matrix,
    y: np.ndarray,
    cfg: ModelConfig,
    *,
    C: float = 1.0,
    penalty: str = "l2",
) -> object:
    lr = LogisticRegression(
        C=C,
        penalty=penalty,
        solver=cfg.lr_solver,
        max_iter=cfg.lr_max_iter,
        class_weight=cfg.lr_class_weight,
        n_jobs=1,
    )
    lr.fit(X, y)
    return _maybe_calibrate(lr, X, y, cfg)


def train_linear_svm(
    X: csr_matrix,
    y: np.ndarray,
    cfg: ModelConfig,
    *,
    C: float = 1.0,
) -> object:
    svm = LinearSVC(
        C=C,
        loss=cfg.svm_loss,
        class_weight=cfg.svm_class_weight,
    )
    svm.fit(X, y)
    # Calibrate to get predict_proba
    return _maybe_calibrate(svm, X, y, cfg)


def tune_linear_models(
    X: csr_matrix,
    y: np.ndarray,
    cfg: ModelConfig,
    *,
    scoring: str = "f1",
    cv: int = 5,
    refit: bool = True,
) -> Dict[str, object]:
    """
    Grid-searches LR and LinearSVC and returns best estimators (optionally calibrated afterward).
    """
    results: Dict[str, object] = {}

    # Logistic Regression tuning
    lr = LogisticRegression(
        solver=cfg.lr_solver,
        max_iter=cfg.lr_max_iter,
        class_weight=cfg.lr_class_weight,
        n_jobs=1,
    )
    lr_grid = {
        "C": list(cfg.lr_C_grid),
        "penalty": ["l2", "l1"],
    }
    lr_search = GridSearchCV(lr, lr_grid, scoring=scoring, cv=cv, n_jobs=-1, refit=refit)
    lr_search.fit(X, y)
    best_lr = lr_search.best_estimator_
    results["logreg"] = _maybe_calibrate(best_lr, X, y, cfg)

    # Linear SVM tuning
    svm = LinearSVC(class_weight=cfg.svm_class_weight)
    svm_grid = {
        "C": list(cfg.svm_C_grid),
        "loss": [cfg.svm_loss],
    }
    svm_search = GridSearchCV(svm, svm_grid, scoring=scoring, cv=cv, n_jobs=-1, refit=refit)
    svm_search.fit(X, y)
    best_svm = svm_search.best_estimator_
    results["linear_svm"] = _maybe_calibrate(best_svm, X, y, cfg)

    return results


def predict_scores(model: object, X: csr_matrix) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Returns:
      - y_pred (hard labels)
      - y_prob (probabilities for class 1 if available, else None)
    """
    y_pred = model.predict(X)

    y_prob: Optional[np.ndarray] = None
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            y_prob = proba[:, 1]
    return y_pred, y_prob
