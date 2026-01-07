# experiments/robustness.py
from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config import DatasetConfig, FeatureConfig, ModelConfig, TextPreprocessConfig
from src.preprocessing import TextPreprocessor, load_fake_true_dataset, load_liar_dataset, load_liar_splits, build_clean_columns
from src.features import FeatureBuilder
from src.models import train_linear_svm
from src.evaluation import compute_metrics
from src.utils import ensure_dir, get_logger, save_json, seed_everything, timer


def stratified_split(df: pd.DataFrame, seed: int, test_size: float):
    from sklearn.model_selection import train_test_split
    y = df["label"].to_numpy(dtype=int)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=y)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def truncate_text(s: str, max_chars: int) -> str:
    if not isinstance(s, str):
        return ""
    return s[:max_chars]


def introduce_typos(text: str, prob: float = 0.02) -> str:
    """
    Simple perturbation: random character swaps/deletions with small probability.
    """
    if not isinstance(text, str) or len(text) < 5:
        return "" if not isinstance(text, str) else text

    chars = list(text)
    i = 0
    while i < len(chars) - 1:
        if random.random() < prob and chars[i].isalpha() and chars[i + 1].isalpha():
            # swap adjacent letters
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
            i += 2
        elif random.random() < prob / 2:
            # delete a char
            del chars[i]
        else:
            i += 1
    return "".join(chars)


def make_variant(df: pd.DataFrame, variant: str, trunc_chars: int | None = None, typo_prob: float | None = None) -> pd.DataFrame:
    """
    Creates a test-set variant without refitting preprocessing.
    Assumes df already has *_clean columns.
    """
    v = df.copy()

    if variant == "title_only":
        v["body_clean"] = ""
        v["combined_clean"] = v["title_clean"].fillna("")
    elif variant == "body_only":
        v["title_clean"] = ""
        v["combined_clean"] = v["body_clean"].fillna("")
    elif variant.startswith("truncate_") and trunc_chars is not None:
        v["body_clean"] = v["body_clean"].apply(lambda x: truncate_text(x, trunc_chars))
        v["combined_clean"] = (v["title_clean"].fillna("") + "\n" + v["body_clean"].fillna("")).str.strip()
    elif variant.startswith("typos_") and typo_prob is not None:
        v["body_clean"] = v["body_clean"].apply(lambda x: introduce_typos(x, prob=typo_prob))
        v["combined_clean"] = (v["title_clean"].fillna("") + "\n" + v["body_clean"].fillna("")).str.strip()

    return v


def main():
    ap = argparse.ArgumentParser(description="Robustness tests for a linear SVM fake-news classifier.")
    ap.add_argument("--fake_csv", type=str, default=str(ROOT / "data/raw/Fake.csv"))
    ap.add_argument("--true_csv", type=str, default=str(ROOT / "data/raw/True.csv"))
    ap.add_argument("--liar_train", type=str, default=str(ROOT / "data/raw/train.tsv"))
    ap.add_argument("--liar_valid", type=str, default=str(ROOT / "data/raw/valid.tsv"))
    ap.add_argument("--liar_test", type=str, default=str(ROOT / "data/raw/test.tsv"))
    ap.add_argument("--dataset", type=str, choices=["fake_true", "liar", "combined"], default="fake_true")
    ap.add_argument("--artifacts_dir", type=str, default=str(ROOT / "artifacts/robustness"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--C_svm", type=float, default=1.0)
    ap.add_argument("--no_calibration", action="store_true")
    args = ap.parse_args()

    logger = get_logger("robustness")
    seed_everything(args.seed)
    random.seed(args.seed)
    out_dir = ensure_dir(args.artifacts_dir)

    ds_cfg = DatasetConfig()
    tp = TextPreprocessor(TextPreprocessConfig())
    feat_cfg = FeatureConfig()
    model_cfg = ModelConfig(calibrate=not args.no_calibration)

    with timer("load data", logger):
        if args.dataset == "fake_true":
            df = load_fake_true_dataset(args.fake_csv, args.true_csv, dataset_cfg=ds_cfg)
            df = build_clean_columns(df, tp=tp, dataset_cfg=ds_cfg)
            train_df, test_df = stratified_split(df, args.seed, args.test_size)
        elif args.dataset == "liar":
            train_df, _, test_df = load_liar_splits(
                args.liar_train,
                args.liar_valid,
                args.liar_test,
                dataset_cfg=ds_cfg,
            )
            train_df = build_clean_columns(train_df, tp=tp, dataset_cfg=ds_cfg)
            test_df = build_clean_columns(test_df, tp=tp, dataset_cfg=ds_cfg)
        else:
            df_ft = load_fake_true_dataset(args.fake_csv, args.true_csv, dataset_cfg=ds_cfg)
            df_liar = load_liar_dataset(args.liar_train, args.liar_valid, args.liar_test, dataset_cfg=ds_cfg)
            df = pd.concat([df_ft, df_liar], ignore_index=True)
            df = build_clean_columns(df, tp=tp, dataset_cfg=ds_cfg)
            train_df, test_df = stratified_split(df, args.seed, args.test_size)

    y_train = train_df["label"].to_numpy(dtype=int)
    y_test = test_df["label"].to_numpy(dtype=int)

    fb = FeatureBuilder(feat_cfg)
    X_train = fb.fit_transform(train_df)

    with timer("train Linear SVM", logger):
        model = train_linear_svm(X_train, y_train, model_cfg, C=args.C_svm)

    variants = [
        ("clean", None, None),
        ("title_only", None, None),
        ("body_only", None, None),
        ("truncate_500", 500, None),
        ("truncate_1000", 1000, None),
        ("truncate_2000", 2000, None),
        ("typos_0.01", None, 0.01),
        ("typos_0.02", None, 0.02),
    ]

    rows = []

    for name, trunc_chars, typo_prob in variants:
        vdf = make_variant(test_df, name, trunc_chars=trunc_chars, typo_prob=typo_prob)
        Xv = fb.transform(vdf)
        y_pred = model.predict(Xv)

        # For AUC/PR, use probabilities if available else decision_function
        y_score = None
        if hasattr(model, "predict_proba"):
            y_score = model.predict_proba(Xv)[:, 1]
        elif hasattr(model, "decision_function"):
            y_score = model.decision_function(Xv)

        res = compute_metrics(y_test, y_pred, y_score)
        rows.append({"variant": name, "n": int(len(vdf)), **res.metrics})

        logger.info("Variant=%s metrics=%s", name, res.metrics)

    results_df = pd.DataFrame(rows).sort_values("f1", ascending=False)
    results_df.to_csv(out_dir / "robustness_results.csv", index=False)

    save_json(out_dir / "robustness_summary.json", {
        "dataset": args.dataset,
        "seed": args.seed,
        "test_size": args.test_size,
        "calibrated": not args.no_calibration,
        "results_path": str(out_dir / "robustness_results.csv"),
    })

    logger.info("Saved: %s", out_dir / "robustness_results.csv")


if __name__ == "__main__":
    main()
