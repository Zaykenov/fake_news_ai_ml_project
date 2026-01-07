# src/preprocessing.py
from __future__ import annotations

import csv
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import pandas as pd

from .config import DatasetConfig, TextPreprocessConfig
from .utils import collapse_whitespace, normalize_label_str, parse_date_series, read_csv_robust


_HTML_RE = re.compile(r"<.*?>")
_URL_RE = re.compile(r"(https?://\S+|www\.\S+)", flags=re.IGNORECASE)
_EMAIL_RE = re.compile(r"\b[\w\.-]+@[\w\.-]+\.\w+\b")
_NON_WORD_KEEP_APOS_RE = re.compile(r"[^a-zA-Z0-9\s']+")


def _try_spacy():
    try:
        import spacy  # type: ignore
        return spacy
    except Exception:
        return None


@dataclass
class TextPreprocessor:
    cfg: TextPreprocessConfig

    def clean_text(self, text: Any) -> str:
        if text is None or (isinstance(text, float) and pd.isna(text)):
            return ""

        s = str(text)

        if self.cfg.normalize_unicode:
            s = unicodedata.normalize("NFKC", s)

        if self.cfg.strip_html:
            s = _HTML_RE.sub(" ", s)

        if self.cfg.strip_urls:
            s = _URL_RE.sub(" ", s)

        if self.cfg.strip_emails:
            s = _EMAIL_RE.sub(" ", s)

        if self.cfg.lowercase:
            s = s.lower()

        if self.cfg.remove_punctuation:
            # Keep apostrophes by default (e.g., "don't")
            s = _NON_WORD_KEEP_APOS_RE.sub(" ", s)

        if self.cfg.remove_numbers:
            s = re.sub(r"\d+", " ", s)

        if self.cfg.collapse_whitespace:
            s = collapse_whitespace(s)

        if self.cfg.lemmatize:
            s = self._lemmatize_spacy(s)

        return s

    def _lemmatize_spacy(self, text: str) -> str:
        spacy = _try_spacy()
        if spacy is None:
            return text
        # lightweight pipeline
        try:
            nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
        except Exception:
            # If model isn't installed, skip lemmatization.
            return text
        doc = nlp(text)
        lemmas = [t.lemma_ for t in doc if not t.is_space]
        return " ".join(lemmas)


def load_fake_true_dataset(
    fake_csv: Union[str, Path],
    true_csv: Union[str, Path],
    *,
    dataset_cfg: DatasetConfig = DatasetConfig(),
    sep: Optional[str] = None,
) -> pd.DataFrame:
    """
    Loads the common Kaggle Fake/True dataset (two CSV files).
    Expected columns: title, text, subject, date
    """
    fake_df = read_csv_robust(fake_csv, sep=sep)
    true_df = read_csv_robust(true_csv, sep=sep)

    # Standardize columns
    for df in (fake_df, true_df):
        df.columns = [c.strip().lower() for c in df.columns]

    required = set(dataset_cfg.fake_true_cols)
    for name, df in [("fake", fake_df), ("true", true_df)]:
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in {name} CSV: {sorted(missing)}")

    fake_df = fake_df[list(dataset_cfg.fake_true_cols)].copy()
    true_df = true_df[list(dataset_cfg.fake_true_cols)].copy()

    fake_df["label"] = dataset_cfg.label_fake
    true_df["label"] = dataset_cfg.label_real

    df = pd.concat([fake_df, true_df], axis=0, ignore_index=True)
    df["source_dataset"] = "fake_true"
    df["id"] = None
    df["speaker"] = None
    df["context"] = None
    df["party"] = None

    # Parse date if present
    df["date"] = parse_date_series(df["date"])

    return df


def load_liar_dataset(
    train_csv: Union[str, Path],
    valid_csv: Union[str, Path],
    test_csv: Union[str, Path],
    *,
    dataset_cfg: DatasetConfig = DatasetConfig(),
    sep: Optional[str] = None,
) -> pd.DataFrame:
    """
    Loads LIAR dataset splits and merges them. Assumes 14 columns in the order specified.

    Binary mapping:
      positive (REAL): true, mostly-true, half-true
      negative (FAKE): barely-true, false, pants-fire
    """
    train_df, valid_df, test_df = load_liar_splits(
        train_csv,
        valid_csv,
        test_csv,
        dataset_cfg=dataset_cfg,
        sep=sep,
    )
    return pd.concat([train_df, valid_df, test_df], axis=0, ignore_index=True)


def _load_liar_split(
    path: Union[str, Path],
    split: str,
    *,
    dataset_cfg: DatasetConfig,
    sep: Optional[str],
) -> pd.DataFrame:
    sep = sep or "\t"
    if sep == "\t":
        df = pd.read_csv(
            path,
            sep=sep,
            header=None,
            quoting=csv.QUOTE_NONE,
            engine="python",
        )
    else:
        df = read_csv_robust(path, sep=sep)
    # If header row exists, keep as-is; otherwise rename by position.
    if df.shape[1] != dataset_cfg.liar_num_columns:
        if sep == "\t":
            df2 = pd.read_csv(
                path,
                sep=sep,
                header=0,
                quoting=csv.QUOTE_NONE,
                engine="python",
            )
        else:
            df2 = read_csv_robust(path, sep=sep)
        if df2.shape[1] == dataset_cfg.liar_num_columns:
            df = df2
        else:
            raise ValueError(
                f"Expected {dataset_cfg.liar_num_columns} columns for LIAR, got {df.shape[1]} in {path}"
            )

    df = df.copy()
    df.columns = [
        "id",
        "liar_label",
        "statement",
        "subject",
        "speaker",
        "speaker_job",
        "state",
        "party",
        "barely_true_counts",
        "false_counts",
        "half_true_counts",
        "mostly_true_counts",
        "pants_on_fire_counts",
        "context",
    ]
    df["split"] = split
    return df


def _map_liar_to_binary(df: pd.DataFrame, dataset_cfg: DatasetConfig) -> pd.DataFrame:
    lab = df["liar_label"].map(normalize_label_str)
    pos = set(dataset_cfg.liar_binary_positive)
    neg = set(dataset_cfg.liar_binary_negative)

    def _map_binary(x: str) -> Optional[int]:
        if x in pos:
            return dataset_cfg.label_real
        if x in neg:
            return dataset_cfg.label_fake
        return None

    out = df.copy()
    out["label"] = lab.map(_map_binary)
    out = out.dropna(subset=["label"]).copy()
    out["label"] = out["label"].astype(int)
    return out


def _liar_to_common_schema(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reduce LIAR to statement-only text and strip metadata that can leak prior credibility.
    """
    return pd.DataFrame({
        "id": df["id"],
        "title": None,
        "text": df["statement"],
        "subject": None,
        "date": pd.NaT,
        "label": df["label"],
        "source_dataset": "liar",
        "speaker": None,
        "context": None,
        "party": None,
        "split": df["split"],
    })


def load_liar_splits(
    train_csv: Union[str, Path],
    valid_csv: Union[str, Path],
    test_csv: Union[str, Path],
    *,
    dataset_cfg: DatasetConfig = DatasetConfig(),
    sep: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Loads LIAR splits and returns them in the common schema using statement text only.
    """
    train_df = _load_liar_split(train_csv, "train", dataset_cfg=dataset_cfg, sep=sep)
    valid_df = _load_liar_split(valid_csv, "valid", dataset_cfg=dataset_cfg, sep=sep)
    test_df = _load_liar_split(test_csv, "test", dataset_cfg=dataset_cfg, sep=sep)

    train_df = _map_liar_to_binary(train_df, dataset_cfg)
    valid_df = _map_liar_to_binary(valid_df, dataset_cfg)
    test_df = _map_liar_to_binary(test_df, dataset_cfg)

    return (
        _liar_to_common_schema(train_df),
        _liar_to_common_schema(valid_df),
        _liar_to_common_schema(test_df),
    )


def build_clean_columns(
    df: pd.DataFrame,
    *,
    tp: TextPreprocessor,
    dataset_cfg: DatasetConfig = DatasetConfig(),
) -> pd.DataFrame:
    """
    Adds:
      - title_clean
      - body_clean
      - subject_clean
      - combined_clean
    Uses conservative defaults and retains original columns.
    """
    df = df.copy()
    if "title" not in df.columns:
        df["title"] = None
    if "text" not in df.columns:
        df["text"] = None
    if "subject" not in df.columns:
        df["subject"] = None

    df["title_clean"] = df["title"].apply(tp.clean_text)
    df["body_clean"] = df["text"].apply(tp.clean_text)
    df["subject_clean"] = df["subject"].apply(tp.clean_text)

    # Combined text (title + body) helps when one is missing / short
    df["combined_clean"] = (df["title_clean"].fillna("") + "\n" + df["body_clean"].fillna("")).str.strip()
    df["combined_clean"] = df["combined_clean"].apply(collapse_whitespace)

    # Apply minimum-length filter
    if tp.cfg.min_chars > 0:
        df = df[df["combined_clean"].str.len() >= tp.cfg.min_chars].copy()

    return df
