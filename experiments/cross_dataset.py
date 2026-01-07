# experiments/cross_dataset.py
from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(ROOT))

from src.config import DatasetConfig, FeatureConfig, ModelConfig, TextPreprocessConfig
from src.preprocessing import TextPreprocessor, load_fake_true_dataset, load_liar_splits, build_clean_columns
from src.features import FeatureBuilder, feature_config_for_liar
from src.models import train_logistic_regression, train_linear_svm
from src.evaluation import compute_metrics
from src.explainability import top_features_linear
from src.utils import ensure_dir, get_logger, save_json, seed_everything, timer


def get_score(model: object, X) -> Optional[np.ndarray]:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)
        if proba.ndim == 2 and proba.shape[1] >= 2:
            return proba[:, 1]
    if hasattr(model, "decision_function"):
        return model.decision_function(X)
    return None


def build_error_report(
    df: pd.DataFrame,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    score: Optional[np.ndarray],
    out_path: Path,
    *,
    top_k: int = 50,
) -> Dict[str, Any]:
    mask = y_true != y_pred
    n_errors = int(mask.sum())
    n_false_positive = int(((y_pred == 1) & (y_true == 0)).sum())
    n_false_negative = int(((y_pred == 0) & (y_true == 1)).sum())

    lengths = df["combined_clean"].fillna("").astype(str).str.len()
    err_lengths = lengths[mask]
    ok_lengths = lengths[~mask]

    stats = {
        "n_errors": n_errors,
        "n_false_positive": n_false_positive,
        "n_false_negative": n_false_negative,
        "mean_len_error": float(err_lengths.mean()) if n_errors else 0.0,
        "mean_len_correct": float(ok_lengths.mean()) if len(ok_lengths) else 0.0,
    }

    cols = [c for c in ["id", "split", "source_dataset", "title", "text", "combined_clean"] if c in df.columns]
    err_df = df.loc[mask, cols].copy()
    err_df["y_true"] = y_true[mask]
    err_df["y_pred"] = y_pred[mask]

    if score is not None:
        err_df["score"] = score[mask]
        if np.nanmin(score) >= 0.0 and np.nanmax(score) <= 1.0:
            err_df["confidence"] = np.abs(err_df["score"] - 0.5)
        else:
            err_df["confidence"] = np.abs(err_df["score"])
        err_df = err_df.sort_values("confidence", ascending=False)
    else:
        err_df["score"] = np.nan
        err_df["confidence"] = np.nan

    err_df["error_type"] = np.where(err_df["y_pred"] == 1, "false_positive", "false_negative")
    err_df = err_df.head(top_k)

    ensure_dir(out_path.parent)
    err_df.to_csv(out_path, index=False)
    return {**stats, "errors_path": str(out_path)}


def train_eval_direction(
    name: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feat_cfg: FeatureConfig,
    model_cfg: ModelConfig,
    *,
    models: List[str],
    C_lr: float,
    C_svm: float,
    top_k_features: int,
    errors_k: int,
    out_dir: Path,
    logger,
) -> Dict[str, Any]:
    results: Dict[str, Any] = {}

    fb = FeatureBuilder(feat_cfg)
    with timer(f"fit_transform features ({name})", logger):
        X_train = fb.fit_transform(train_df)
    with timer(f"transform features ({name})", logger):
        X_test = fb.transform(test_df)

    y_train = train_df["label"].to_numpy(dtype=int)
    y_test = test_df["label"].to_numpy(dtype=int)
    feature_names = fb.get_feature_names()

    for model_name in models:
        if model_name == "logreg":
            model = train_logistic_regression(X_train, y_train, model_cfg, C=C_lr, penalty="l2")
        elif model_name == "linear_svm":
            model = train_linear_svm(X_train, y_train, model_cfg, C=C_svm)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        y_pred = model.predict(X_test)
        score = get_score(model, X_test)
        res = compute_metrics(y_test, y_pred, score)

        tf = None
        try:
            tf = top_features_linear(model, feature_names, top_k=top_k_features)
        except Exception:
            tf = None

        errors_path = out_dir / f"errors_{name}_{model_name}.csv"
        error_stats = build_error_report(
            test_df,
            y_test,
            y_pred,
            score,
            errors_path,
            top_k=errors_k,
        )

        model_out: Dict[str, Any] = {
            "metrics": res.metrics,
            "confusion_matrix": res.confusion.tolist(),
            "error_analysis": error_stats,
        }
        if tf is not None:
            model_out["top_features"] = {
                "positive_real": tf.positive,
                "negative_fake": tf.negative,
            }

        results[model_name] = model_out
        logger.info("%s/%s metrics: %s", name, model_name, res.metrics)

    return {
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "feature_dim": int(X_train.shape[1]),
        "models": results,
    }


def write_notes(path: Path, summary: Dict[str, Any]) -> None:
    lines: List[str] = []
    lines.append("Cross-dataset generalization")
    for direction, entry in summary["directions"].items():
        lines.append(f"{direction}: train={entry['train_dataset']} test={entry['test_dataset']}")
        for model_name, model_res in entry["models"].items():
            m = model_res["metrics"]
            lines.append(
                f"- {model_name}: accuracy={m.get('accuracy', 0.0):.4f} "
                f"precision={m.get('precision', 0.0):.4f} "
                f"recall={m.get('recall', 0.0):.4f} "
                f"f1={m.get('f1', 0.0):.4f} "
                f"roc_auc={m.get('roc_auc', float('nan')):.4f}"
            )

    lines.append("")
    lines.append("Ethical limitations")
    for note in summary["ethical_limitations"]:
        lines.append(f"- {note}")

    ensure_dir(path.parent)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-dataset generalization between Fake/True and LIAR.")
    ap.add_argument("--fake_csv", type=str, default=str(ROOT / "data/raw/Fake.csv"))
    ap.add_argument("--true_csv", type=str, default=str(ROOT / "data/raw/True.csv"))
    ap.add_argument("--liar_train", type=str, default=str(ROOT / "data/raw/train.tsv"))
    ap.add_argument("--liar_valid", type=str, default=str(ROOT / "data/raw/valid.tsv"))
    ap.add_argument("--liar_test", type=str, default=str(ROOT / "data/raw/test.tsv"))
    ap.add_argument("--artifacts_dir", type=str, default=str(ROOT / "artifacts/cross_dataset"))
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--direction", choices=["articles_to_liar", "liar_to_articles", "both"], default="both")
    ap.add_argument("--models", choices=["logreg", "linear_svm", "both"], default="logreg")
    ap.add_argument("--C_lr", type=float, default=1.0)
    ap.add_argument("--C_svm", type=float, default=1.0)
    ap.add_argument("--no_calibration", action="store_true")
    ap.add_argument("--top_k_features", type=int, default=30)
    ap.add_argument("--errors_k", type=int, default=50)
    args = ap.parse_args()

    logger = get_logger("cross_dataset")
    seed_everything(args.seed)

    out_dir = ensure_dir(args.artifacts_dir)

    ds_cfg = DatasetConfig()
    tp = TextPreprocessor(TextPreprocessConfig())
    feat_cfg = FeatureConfig()
    feat_cfg_liar = feature_config_for_liar(feat_cfg)
    model_cfg = ModelConfig(calibrate=not args.no_calibration)

    with timer("load Fake/True", logger):
        fake_true_df = load_fake_true_dataset(args.fake_csv, args.true_csv, dataset_cfg=ds_cfg)
        fake_true_df = build_clean_columns(fake_true_df, tp=tp, dataset_cfg=ds_cfg)

    with timer("load LIAR splits", logger):
        liar_train_df, liar_valid_df, liar_test_df = load_liar_splits(
            args.liar_train,
            args.liar_valid,
            args.liar_test,
            dataset_cfg=ds_cfg,
        )
        liar_train_df = build_clean_columns(liar_train_df, tp=tp, dataset_cfg=ds_cfg)
        liar_valid_df = build_clean_columns(liar_valid_df, tp=tp, dataset_cfg=ds_cfg)
        liar_test_df = build_clean_columns(liar_test_df, tp=tp, dataset_cfg=ds_cfg)
        liar_train_full = pd.concat([liar_train_df, liar_valid_df], ignore_index=True)

    model_list = ["logreg", "linear_svm"] if args.models == "both" else [args.models]

    summary: Dict[str, Any] = {
        "directions": {},
        "config": {
            "models": model_list,
            "C_lr": args.C_lr,
            "C_svm": args.C_svm,
            "calibrated": not args.no_calibration,
        },
        "ethical_limitations": [
            "Lexical cues can correlate with style or topic rather than factuality.",
            "Dataset shift can drive errors when training on articles and testing on claims.",
            "Models ignore evidence and context, so they do not verify claims.",
            "Prediction errors can disproportionately affect speakers or groups.",
        ],
    }

    if args.direction in ("articles_to_liar", "both"):
        entry = train_eval_direction(
            "articles_to_liar",
            train_df=fake_true_df,
            test_df=liar_test_df,
            feat_cfg=feat_cfg,
            model_cfg=model_cfg,
            models=model_list,
            C_lr=args.C_lr,
            C_svm=args.C_svm,
            top_k_features=args.top_k_features,
            errors_k=args.errors_k,
            out_dir=out_dir,
            logger=logger,
        )
        entry["train_dataset"] = "fake_true"
        entry["test_dataset"] = "liar_test"
        summary["directions"]["articles_to_liar"] = entry

    if args.direction in ("liar_to_articles", "both"):
        entry = train_eval_direction(
            "liar_to_articles",
            train_df=liar_train_full,
            test_df=fake_true_df,
            feat_cfg=feat_cfg_liar,
            model_cfg=model_cfg,
            models=model_list,
            C_lr=args.C_lr,
            C_svm=args.C_svm,
            top_k_features=args.top_k_features,
            errors_k=args.errors_k,
            out_dir=out_dir,
            logger=logger,
        )
        entry["train_dataset"] = "liar_train+valid"
        entry["test_dataset"] = "fake_true"
        summary["directions"]["liar_to_articles"] = entry

    save_json(out_dir / "cross_dataset_results.json", summary)
    write_notes(out_dir / "analysis_notes.md", summary)
    logger.info("Saved: %s", out_dir / "cross_dataset_results.json")
    logger.info("Saved: %s", out_dir / "analysis_notes.md")


if __name__ == "__main__":
    main()
