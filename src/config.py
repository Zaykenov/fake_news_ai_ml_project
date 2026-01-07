# src/config.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple, Optional


@dataclass(frozen=True)
class Paths:
    """
    Centralized project paths.

    Adjust base_dir if you run scripts from a different working directory.
    """
    base_dir: Path = Path(__file__).resolve().parents[1]  # .../fake-news-detection
    data_dir: Path = base_dir / "data"
    raw_dir: Path = data_dir / "raw"
    processed_dir: Path = data_dir / "processed"
    artifacts_dir: Path = base_dir / "artifacts"
    reports_dir: Path = base_dir / "reports"


@dataclass(frozen=True)
class TextPreprocessConfig:
    """
    Conservative defaults to reduce leakage and noise without overfitting.
    """
    lowercase: bool = True
    strip_html: bool = True
    strip_urls: bool = True
    strip_emails: bool = True
    normalize_unicode: bool = True
    collapse_whitespace: bool = True

    # Optional normalization knobs (keep off by default for reproducibility and speed)
    remove_punctuation: bool = False
    remove_numbers: bool = False

    # If True, attempts spaCy lemmatization (falls back gracefully if unavailable).
    lemmatize: bool = False

    # Drop very short docs that are often junk or metadata.
    min_chars: int = 30


@dataclass(frozen=True)
class DatasetConfig:
    """
    Label conventions:
    - Binary: 1 = REAL/TRUE, 0 = FAKE
    """
    label_real: int = 1
    label_fake: int = 0

    # LIAR label mapping for binary tasks
    liar_binary_positive: Tuple[str, ...] = ("true", "mostly-true", "half-true")
    liar_binary_negative: Tuple[str, ...] = ("barely-true", "false", "pants-fire")

    # Column names for Fake/True dataset
    fake_true_cols: Tuple[str, ...] = ("title", "text", "subject", "date")

    # LIAR columns by position (1-indexed in the prompt; 0-indexed in code)
    # 0: id, 1: label, 2: statement, 3: subject(s), 4: speaker, 5: job title,
    # 6: state, 7: party, 8-12: history counts, 13: context
    liar_num_columns: int = 14


@dataclass(frozen=True)
class FeatureViewConfig:
    """
    One "view" == one TfidfVectorizer applied to one text column.
    """
    name: str
    column: str
    analyzer: str  # "word" or "char"
    ngram_range: Tuple[int, int]
    max_features: Optional[int] = None
    min_df: int = 2
    max_df: float = 0.95
    sublinear_tf: bool = True


@dataclass(frozen=True)
class FeatureConfig:
    """
    Multi-view TF-IDF:
    - word ngrams for semantics
    - char ngrams for robustness (typos, obfuscation)
    """
    views: Tuple[FeatureViewConfig, ...] = field(default_factory=lambda: (
        FeatureViewConfig(
            name="word_body",
            column="body_clean",
            analyzer="word",
            ngram_range=(1, 2),
            max_features=200_000,
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        ),
        FeatureViewConfig(
            name="char_body",
            column="body_clean",
            analyzer="char",
            ngram_range=(3, 5),
            max_features=300_000,
            min_df=3,
            max_df=0.95,
            sublinear_tf=True,
        ),
        FeatureViewConfig(
            name="word_title",
            column="title_clean",
            analyzer="word",
            ngram_range=(1, 2),
            max_features=50_000,
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
        ),
    ))

    # Optional: include "subject" in features (toggle in your experiment scripts)
    enable_subject_view: bool = False


@dataclass(frozen=True)
class ModelConfig:
    """
    Model defaults (good baselines).
    """
    # Logistic Regression (sparse-friendly)
    lr_solver: str = "liblinear"  # stable for L2, L1
    lr_max_iter: int = 2000
    lr_class_weight: str = "balanced"

    # Linear SVM
    svm_loss: str = "squared_hinge"
    svm_class_weight: str = "balanced"

    # Calibration for decision-ready probabilities
    calibrate: bool = True
    calibration_method: str = "sigmoid"  # "sigmoid" or "isotonic"
    calibration_cv: int = 3

    # Hyperparameter search grids (use in experiments)
    lr_C_grid: Tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0)
    svm_C_grid: Tuple[float, ...] = (0.1, 0.3, 1.0, 3.0, 10.0)


@dataclass(frozen=True)
class TrainConfig:
    """
    Train/val/test controls.
    """
    seed: int = 42
    test_size: float = 0.2
    val_size: float = 0.2  # fraction of remaining after test split
    stratify: bool = True

    # If True, drops rows with missing body/title after cleaning
    drop_empty: bool = True
