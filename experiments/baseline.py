# experiments/baseline.py
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config import DatasetConfig, FeatureConfig, ModelConfig, TextPreprocessConfig, TrainConfig, Paths
from src.preprocessing import (
    TextPreprocessor,
    load_fake_true_dataset,
    load_liar_dataset,
    build_clean_columns,
)
from src.features import FeatureBuilder
from src.models import tune_linear_models, predict_scores
from src.evaluation import compute_metrics
from src.explainability import top_features_linear
from src.utils import ensure_dir, get_logger, save_json, seed_everything, timer


def stratified_split_df(df: pd.DataFrame, y: np.ndarray, seed: int, test_size: float):
    from sklearn.model_selection import train_test_split
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=y,
    )
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="Baseline experiment: multi-view TF-IDF + tuned LR and Linear SVM.")
    ap.add_argument("--fake_csv", type=str, default=str(ROOT / "data/raw/Fake.csv"))
    ap.add_argument("--true_csv", type=str, default=str(ROOT / "data/raw/True.csv"))
    ap.add_argument("--liar_train", type=str, default=str(ROOT / "data/raw/liar_train.csv"))
    ap.add_argument("--liar_valid", type=str, default=str(ROOT / "data/raw/liar_valid.csv"))
    ap.add_argument("--liar_test", type=str, default=str(ROOT / "data/raw/liar_test.csv"))
    ap.add_argument("--dataset", type=str, choices=["fake_true", "liar", "combined"], default="fake_true")
    ap.add_argument("--artifacts_dir", type=str, default=str(ROOT / "artifacts/baseline"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--top_k", type=int, default=30)
    ap.add_argument("--drop_duplicates", action="store_true", help="Drop exact duplicates using combined_clean.")
    ap.add_argument("--cv", type=int, default=3, help="CV folds for grid search (lower saves memory).")
    ap.add_argument("--n_jobs", type=int, default=1, help="Parallel jobs for grid search.")
    args = ap.parse_args()

    logger = get_logger("baseline")
    seed_everything(args.seed)

    artifacts_dir = ensure_dir(args.artifacts_dir)

    ds_cfg = DatasetConfig()
    tp_cfg = TextPreprocessConfig()
    tr_cfg = TrainConfig(seed=args.seed, test_size=args.test_size)

    logger.info("Loading dataset: %s", args.dataset)
    with timer("load data", logger):
        if args.dataset == "fake_true":
            df = load_fake_true_dataset(args.fake_csv, args.true_csv, dataset_cfg=ds_cfg)
        elif args.dataset == "liar":
            df = load_liar_dataset(args.liar_train, args.liar_valid, args.liar_test, dataset_cfg=ds_cfg)
        else:
            df_ft = load_fake_true_dataset(args.fake_csv, args.true_csv, dataset_cfg=ds_cfg)
            df_liar = load_liar_dataset(args.liar_train, args.liar_valid, args.liar_test, dataset_cfg=ds_cfg)
            df = pd.concat([df_ft, df_liar], ignore_index=True)

    tp = TextPreprocessor(tp_cfg)
    with timer("build clean columns", logger):
        df = build_clean_columns(df, tp=tp, dataset_cfg=ds_cfg)

    if args.drop_duplicates:
        before = len(df)
        df = df.drop_duplicates(subset=["combined_clean"]).reset_index(drop=True)
        logger.info("Dropped duplicates: %d -> %d", before, len(df))

    y = df["label"].to_numpy(dtype=int)

    train_df, test_df = stratified_split_df(df, y, args.seed, args.test_size)
    y_train = train_df["label"].to_numpy(dtype=int)
    y_test = test_df["label"].to_numpy(dtype=int)

    feat_cfg = FeatureConfig()
    fb = FeatureBuilder(feat_cfg)

    with timer("fit_transform features (train)", logger):
        X_train = fb.fit_transform(train_df)
    with timer("transform features (test)", logger):
        X_test = fb.transform(test_df)

    model_cfg = ModelConfig(calibrate=True)
    with timer("tune linear models", logger):
        models = tune_linear_models(
            X_train,
            y_train,
            model_cfg,
            scoring="f1",
            cv=args.cv,
            n_jobs=args.n_jobs,
        )

    feature_names = fb.get_feature_names()

    summary = {
        "dataset": args.dataset,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "feature_dim": int(X_train.shape[1]),
        "models": {},
        "config": {
            "seed": args.seed,
            "test_size": args.test_size,
            "text_preprocess": tp_cfg,
            "feature_config": feat_cfg,
            "model_config": model_cfg,
        },
    }

    for name, model in models.items():
        y_pred, y_prob = predict_scores(model, X_test)
        res = compute_metrics(y_test, y_pred, y_prob)

        tf = None
        try:
            tf = top_features_linear(model, feature_names, top_k=args.top_k)
        except Exception as e:
            logger.warning("Explainability failed for %s: %s", name, e)

        model_out = {
            "metrics": res.metrics,
            "confusion_matrix": res.confusion.tolist(),
            "classification_report": res.report,
        }
        if tf is not None:
            model_out["top_features"] = {
                "positive_real": tf.positive,
                "negative_fake": tf.negative,
            }

        summary["models"][name] = model_out

        logger.info("%s metrics: %s", name, res.metrics)

    save_json(artifacts_dir / "baseline_results.json", summary)
    logger.info("Saved: %s", artifacts_dir / "baseline_results.json")


if __name__ == "__main__":
    main()
