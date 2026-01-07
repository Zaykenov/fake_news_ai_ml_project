# src/utils.py
from __future__ import annotations

import json
import logging
import os
import random
import re
import time
from contextlib import contextmanager
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Optional, Tuple, Union

import numpy as np
import pandas as pd


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def ensure_dir(path: Union[str, Path]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def get_logger(name: str = "fake_news", level: int = logging.INFO) -> logging.Logger:
    logger = logging.getLogger(name)
    if logger.handlers:
        return logger
    logger.setLevel(level)
    handler = logging.StreamHandler()
    fmt = logging.Formatter("[%(asctime)s] %(levelname)s %(name)s: %(message)s")
    handler.setFormatter(fmt)
    logger.addHandler(handler)
    logger.propagate = False
    return logger


@contextmanager
def timer(msg: str, logger: Optional[logging.Logger] = None):
    t0 = time.time()
    yield
    dt = time.time() - t0
    if logger:
        logger.info("%s in %.2fs", msg, dt)
    else:
        print(f"{msg} in {dt:.2f}s")


def read_csv_robust(
    path: Union[str, Path],
    *,
    sep: Optional[str] = None,
    encoding: Optional[str] = None,
    engine: str = "python",
) -> pd.DataFrame:
    """
    Robust CSV reader. If sep is None, tries common separators.
    """
    path = Path(path)
    if sep is not None:
        return pd.read_csv(path, sep=sep, encoding=encoding, engine=engine)

    # Try comma first; then tab; then pipe
    for candidate in [",", "\t", "|"]:
        try:
            df = pd.read_csv(path, sep=candidate, encoding=encoding, engine=engine)
            # Heuristic: if it's one column only, try next separator
            if df.shape[1] <= 1:
                continue
            return df
        except Exception:
            continue

    # Last resort: let pandas infer
    return pd.read_csv(path, encoding=encoding, engine=engine)


def safe_to_json(obj: Any) -> Any:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, (Path,)):
        return str(obj)
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj


def save_json(path: Union[str, Path], payload: Dict[str, Any]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False, default=safe_to_json)


def parse_date_series(s: pd.Series) -> pd.Series:
    """
    Parses dates robustly; returns pandas datetime with NaT on failures.
    """
    return pd.to_datetime(s, errors="coerce", infer_datetime_format=True)


def normalize_label_str(x: Any) -> str:
    if pd.isna(x):
        return ""
    return str(x).strip().lower()


_whitespace_re = re.compile(r"\s+")
def collapse_whitespace(text: str) -> str:
    return _whitespace_re.sub(" ", text).strip()
