# src/explainability.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.sparse import csr_matrix

from .features import FeatureBuilder


def _unwrap_linear_estimator(model: object) -> object:
    """
    If model is calibrated, attempt to access the underlying linear estimator.
    Works for CalibratedClassifierCV.
    """
    if hasattr(model, "estimator"):
        # sklearn >=1.2 CalibratedClassifierCV has .estimator
        return getattr(model, "estimator")
    if hasattr(model, "base_estimator"):
        return getattr(model, "base_estimator")
    # Older versions: calibrated_classifiers_ contains fitted estimators
    if hasattr(model, "calibrated_classifiers_"):
        ccs = getattr(model, "calibrated_classifiers_")
        if ccs and hasattr(ccs[0], "estimator"):
            return ccs[0].estimator
    return model


@dataclass
class TopFeatures:
    positive: List[Tuple[str, float]]
    negative: List[Tuple[str, float]]


def top_features_linear(
    model: object,
    feature_names: List[str],
    *,
    top_k: int = 30,
) -> TopFeatures:
    """
    Extracts the most influential features for a linear classifier.
    Assumes binary classification with coef_ shape (1, n_features).
    """
    est = _unwrap_linear_estimator(model)
    if not hasattr(est, "coef_"):
        raise ValueError("Model has no coef_. Explainability requires a linear model with coef_.")

    coef = est.coef_
    if coef.ndim == 2:
        coef = coef[0]
    coef = np.asarray(coef).ravel()

    if len(coef) != len(feature_names):
        raise ValueError(f"coef length ({len(coef)}) != feature_names length ({len(feature_names)})")

    idx_sorted = np.argsort(coef)
    neg_idx = idx_sorted[:top_k]
    pos_idx = idx_sorted[-top_k:][::-1]

    negative = [(feature_names[i], float(coef[i])) for i in neg_idx]
    positive = [(feature_names[i], float(coef[i])) for i in pos_idx]
    return TopFeatures(positive=positive, negative=negative)


def explain_instance_linear(
    model: object,
    X_row: csr_matrix,
    feature_names: List[str],
    *,
    top_k: int = 15,
) -> Dict[str, List[Tuple[str, float]]]:
    """
    Provides per-instance contribution scores for a linear model:
      contribution = x_i * w_i
    Only non-zero features are returned, sorted by absolute contribution.
    """
    est = _unwrap_linear_estimator(model)
    if not hasattr(est, "coef_"):
        raise ValueError("Model has no coef_. Instance explanation requires coef_.")

    coef = est.coef_
    if coef.ndim == 2:
        coef = coef[0]
    w = np.asarray(coef).ravel()

    # Get non-zero indices from sparse row
    row = X_row.tocsr()
    nz_idx = row.indices
    nz_data = row.data

    contrib = nz_data * w[nz_idx]
    order = np.argsort(np.abs(contrib))[::-1][:top_k]

    feats = [(feature_names[int(nz_idx[i])], float(contrib[i])) for i in order]
    # Positive supports class 1; negative supports class 0
    pos = [(f, c) for f, c in feats if c >= 0]
    neg = [(f, c) for f, c in feats if c < 0]

    return {"support_real": pos[:top_k], "support_fake": neg[:top_k]}
