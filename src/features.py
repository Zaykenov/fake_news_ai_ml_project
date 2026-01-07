# src/features.py
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, hstack
from sklearn.feature_extraction.text import TfidfVectorizer

from .config import FeatureConfig, FeatureViewConfig


@dataclass
class FeatureBuilder:
    """
    Multi-view TF-IDF feature builder.

    Each view uses its own TfidfVectorizer on a specified text column.
    Resulting sparse matrices are concatenated with hstack in the view order.
    """
    cfg: FeatureConfig
    vectorizers: List[TfidfVectorizer] = None
    view_defs: List[FeatureViewConfig] = None

    def __post_init__(self):
        self.view_defs = list(self.cfg.views)
        self.vectorizers = [self._make_vectorizer(v) for v in self.view_defs]

    @staticmethod
    def _make_vectorizer(v: FeatureViewConfig) -> TfidfVectorizer:
        return TfidfVectorizer(
            analyzer=v.analyzer,
            ngram_range=v.ngram_range,
            max_features=v.max_features,
            min_df=v.min_df,
            max_df=v.max_df,
            sublinear_tf=v.sublinear_tf,
            strip_accents="unicode",
        )

    def fit(self, df: pd.DataFrame) -> "FeatureBuilder":
        for vec, vdef in zip(self.vectorizers, self.view_defs):
            texts = df[vdef.column].fillna("").astype(str).tolist()
            vec.fit(texts)
        return self

    def transform(self, df: pd.DataFrame) -> csr_matrix:
        mats = []
        for vec, vdef in zip(self.vectorizers, self.view_defs):
            texts = df[vdef.column].fillna("").astype(str).tolist()
            mats.append(vec.transform(texts))
        return hstack(mats, format="csr")

    def fit_transform(self, df: pd.DataFrame) -> csr_matrix:
        mats = []
        for vec, vdef in zip(self.vectorizers, self.view_defs):
            texts = df[vdef.column].fillna("").astype(str).tolist()
            mats.append(vec.fit_transform(texts))
        return hstack(mats, format="csr")

    def get_feature_names(self) -> List[str]:
        names: List[str] = []
        for vec, vdef in zip(self.vectorizers, self.view_defs):
            fn = vec.get_feature_names_out()
            names.extend([f"{vdef.name}:{t}" for t in fn])
        return names

    def feature_slices(self) -> List[Tuple[str, slice]]:
        """
        Returns slices into the final concatenated matrix for each view.
        Useful for per-view ablations and analysis.
        """
        slices = []
        start = 0
        for vec, vdef in zip(self.vectorizers, self.view_defs):
            n = len(vec.get_feature_names_out())
            slices.append((vdef.name, slice(start, start + n)))
            start += n
        return slices
