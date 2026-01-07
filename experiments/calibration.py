# experiments/calibration.py
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config import DatasetConfig, FeatureConfig, ModelConfig, TextPreprocessConfig
from src.preprocessing import TextPreprocessor, load_fake_true_dataset, load_liar_dataset, load_liar_splits, build_clean_columns
from src.features import FeatureBuilder, feature_config_for_liar
from src.models import train_logistic_regression, train_linear_svm
from src.evaluation import compute_metrics, find_best_threshold_f1, plot_roc, plot_precision_recall
from src.utils import ensure_dir, get_logger, save_json, seed_everything, timer


def stratified_split(df: pd.DataFrame, seed: int, test_size: float):
    from sklearn.model_selection import train_test_split
    y = df["label"].to_numpy(dtype=int)
    train_df, test_df = train_test_split(df, test_size=test_size, random_state=seed, stratify=y)
    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def get_continuous_score(model, X):
    """
    For AUC/PR: use predict_proba if available; otherwise decision_function.
    """
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        return proba[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def main():
    ap = argparse.ArgumentParser(description="Calibration and thresholding experiment.")
    ap.add_argument("--fake_csv", type=str, default=str(ROOT / "data/raw/Fake.csv"))
    ap.add_argument("--true_csv", type=str, default=str(ROOT / "data/raw/True.csv"))
    ap.add_argument("--liar_train", type=str, default=str(ROOT / "data/raw/train.tsv"))
    ap.add_argument("--liar_valid", type=str, default=str(ROOT / "data/raw/valid.tsv"))
    ap.add_argument("--liar_test", type=str, default=str(ROOT / "data/raw/test.tsv"))
    ap.add_argument("--dataset", type=str, choices=["fake_true", "liar", "combined"], default="fake_true")
    ap.add_argument("--artifacts_dir", type=str, default=str(ROOT / "artifacts/calibration"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--C_lr", type=float, default=1.0)
    ap.add_argument("--C_svm", type=float, default=1.0)
    ap.add_argument("--method", type=str, choices=["sigmoid", "isotonic"], default="sigmoid")
    args = ap.parse_args()

    logger = get_logger("calibration")
    seed_everything(args.seed)
    out_dir = ensure_dir(args.artifacts_dir)

    ds_cfg = DatasetConfig()
    tp = TextPreprocessor(TextPreprocessConfig())
    feat_cfg = FeatureConfig()
    if args.dataset == "liar":
        feat_cfg = feature_config_for_liar(feat_cfg)

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
    X_test = fb.transform(test_df)

    # Uncalibrated configs
    cfg_uncal = ModelConfig(calibrate=False)
    cfg_cal = ModelConfig(calibrate=True, calibration_method=args.method)

    models = {}

    with timer("train LR (uncal)", logger):
        models["lr_uncal"] = train_logistic_regression(X_train, y_train, cfg_uncal, C=args.C_lr, penalty="l2")
    with timer("train LR (cal)", logger):
        models["lr_cal"] = train_logistic_regression(X_train, y_train, cfg_cal, C=args.C_lr, penalty="l2")

    with timer("train SVM (uncal)", logger):
        models["svm_uncal"] = train_linear_svm(X_train, y_train, cfg_uncal, C=args.C_svm)
    with timer("train SVM (cal)", logger):
        models["svm_cal"] = train_linear_svm(X_train, y_train, cfg_cal, C=args.C_svm)

    summary = {"dataset": args.dataset, "method": args.method, "results": {}}

    for name, model in models.items():
        y_pred = model.predict(X_test)
        score = get_continuous_score(model, X_test)
        res = compute_metrics(y_test, y_pred, score)

        entry = {"metrics": res.metrics, "confusion_matrix": res.confusion.tolist()}
        if score is not None:
            best_thr, best_f1 = find_best_threshold_f1(y_test, score)
            entry["best_threshold_f1"] = best_thr
            entry["best_f1_at_threshold"] = best_f1

            # Save plots
            try:
                fig_roc = plot_roc(y_test, score, title=f"ROC - {name}")
                fig_roc.savefig(out_dir / f"roc_{name}.png", dpi=150, bbox_inches="tight")
                fig_pr = plot_precision_recall(y_test, score, title=f"PR - {name}")
                fig_pr.savefig(out_dir / f"pr_{name}.png", dpi=150, bbox_inches="tight")
            except Exception as e:
                logger.warning("Plotting failed for %s: %s", name, e)

        summary["results"][name] = entry
        logger.info("%s metrics: %s", name, res.metrics)

    save_json(out_dir / "calibration_results.json", summary)
    logger.info("Saved: %s", out_dir / "calibration_results.json")


if __name__ == "__main__":
    main()
