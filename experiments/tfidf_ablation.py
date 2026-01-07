# experiments/tfidf_ablation.py
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path
import sys
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config import DatasetConfig, FeatureConfig, FeatureViewConfig, ModelConfig, TextPreprocessConfig
from src.preprocessing import TextPreprocessor, load_fake_true_dataset, load_liar_dataset, load_liar_splits, build_clean_columns
from src.features import FeatureBuilder, feature_config_for_liar
from src.models import train_logistic_regression, train_linear_svm
from src.evaluation import compute_metrics
from src.utils import ensure_dir, get_logger, save_json, seed_everything, timer


def build_feature_cfg(views: list[FeatureViewConfig]) -> FeatureConfig:
    return FeatureConfig(views=tuple(views))


def get_ablation_configs() -> dict[str, FeatureConfig]:
    # Common views (match your src/config.py defaults)
    word_body = FeatureViewConfig(
        name="word_body", column="body_clean", analyzer="word", ngram_range=(1, 2),
        max_features=200_000, min_df=2, max_df=0.95, sublinear_tf=True
    )
    char_body = FeatureViewConfig(
        name="char_body", column="body_clean", analyzer="char", ngram_range=(3, 5),
        max_features=300_000, min_df=3, max_df=0.95, sublinear_tf=True
    )
    word_title = FeatureViewConfig(
        name="word_title", column="title_clean", analyzer="word", ngram_range=(1, 2),
        max_features=50_000, min_df=2, max_df=0.95, sublinear_tf=True
    )

    return {
        "body_word": build_feature_cfg([word_body]),
        "body_char": build_feature_cfg([char_body]),
        "title_word": build_feature_cfg([word_title]),
        "body_word+char": build_feature_cfg([word_body, char_body]),
        "title+body_word": build_feature_cfg([word_title, word_body]),
        "all_views": build_feature_cfg([word_body, char_body, word_title]),
    }


def stratified_split(df: pd.DataFrame, seed: int, test_size: float):
    from sklearn.model_selection import train_test_split
    y = df["label"].to_numpy(dtype=int)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=y)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def main():
    ap = argparse.ArgumentParser(description="TF-IDF ablations over feature views with LR and Linear SVM.")
    ap.add_argument("--fake_csv", type=str, default=str(ROOT / "data/raw/Fake.csv"))
    ap.add_argument("--true_csv", type=str, default=str(ROOT / "data/raw/True.csv"))
    ap.add_argument("--liar_train", type=str, default=str(ROOT / "data/raw/train.tsv"))
    ap.add_argument("--liar_valid", type=str, default=str(ROOT / "data/raw/valid.tsv"))
    ap.add_argument("--liar_test", type=str, default=str(ROOT / "data/raw/test.tsv"))
    ap.add_argument("--dataset", type=str, choices=["fake_true", "liar", "combined"], default="fake_true")
    ap.add_argument("--artifacts_dir", type=str, default=str(ROOT / "artifacts/tfidf_ablation"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--C_lr", type=float, default=1.0)
    ap.add_argument("--C_svm", type=float, default=1.0)
    ap.add_argument("--no_calibration", action="store_true")
    args = ap.parse_args()

    logger = get_logger("tfidf_ablation")
    seed_everything(args.seed)
    out_dir = ensure_dir(args.artifacts_dir)

    ds_cfg = DatasetConfig()
    tp = TextPreprocessor(TextPreprocessConfig())
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

    ablations = get_ablation_configs()
    if args.dataset == "liar":
        ablations = {
            name: cfg for name, cfg in ablations.items()
            if all(view.column == "body_clean" for view in cfg.views)
        }
    rows = []

    for ab_name, feat_cfg in ablations.items():
        logger.info("Ablation: %s", ab_name)
        fb = FeatureBuilder(feat_cfg)
        X_train = fb.fit_transform(train_df)
        X_test = fb.transform(test_df)

        # LR
        lr = train_logistic_regression(X_train, y_train, model_cfg, C=args.C_lr, penalty="l2")
        y_pred, y_prob = (lr.predict(X_test), lr.predict_proba(X_test)[:, 1]) if hasattr(lr, "predict_proba") else (lr.predict(X_test), None)
        res_lr = compute_metrics(y_test, y_pred, y_prob)

        rows.append({
            "ablation": ab_name,
            "model": "logreg",
            "feature_dim": int(X_train.shape[1]),
            **res_lr.metrics
        })

        # SVM
        svm = train_linear_svm(X_train, y_train, model_cfg, C=args.C_svm)
        y_pred, y_prob = (svm.predict(X_test), svm.predict_proba(X_test)[:, 1]) if hasattr(svm, "predict_proba") else (svm.predict(X_test), None)
        res_svm = compute_metrics(y_test, y_pred, y_prob)

        rows.append({
            "ablation": ab_name,
            "model": "linear_svm",
            "feature_dim": int(X_train.shape[1]),
            **res_svm.metrics
        })

    results_df = pd.DataFrame(rows).sort_values(["model", "f1"], ascending=[True, False])
    results_df.to_csv(out_dir / "tfidf_ablation_results.csv", index=False)

    save_json(out_dir / "tfidf_ablation_summary.json", {
        "dataset": args.dataset,
        "seed": args.seed,
        "test_size": args.test_size,
        "calibrated": not args.no_calibration,
        "results_path": str(out_dir / "tfidf_ablation_results.csv"),
    })

    logger.info("Saved: %s", out_dir / "tfidf_ablation_results.csv")


if __name__ == "__main__":
    main()
