"""Train, evaluate, and save a leakage-safe used-car price regression pipeline.

Consumes M1's cleaned CSV (``data/processed/vehicles_clean.csv``), performs a
single deterministic train/test split, fits all preprocessing exclusively on
the training split inside a scikit-learn ``Pipeline``, compares a median
``DummyRegressor`` baseline against a ``RandomForestRegressor`` on the exact
same test rows, and exports the fitted model and test metrics.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.shared.config import load_config, setup_logging

LOG = logging.getLogger(__name__)
MIN_ROWS_REQUIRED = 10


def get_feature_columns(cfg: dict[str, Any]) -> list[str]:
    """Return the agreed ordered feature column list from configuration."""
    return list(cfg["features"]["numeric"]) + list(cfg["features"]["categorical"])


def build_pipeline(cfg: dict[str, Any]) -> Pipeline:
    """Build a fit-on-train-only preprocessing and Random Forest pipeline."""
    num_cols = list(cfg["features"]["numeric"])
    cat_cols = list(cfg["features"]["categorical"])

    preprocess = ColumnTransformer(
        transformers=[
            (
                "num",
                Pipeline([("impute", SimpleImputer(strategy="median"))]),
                num_cols,
            ),
            (
                "cat",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        ("encode", OneHotEncoder(handle_unknown="ignore")),
                    ]
                ),
                cat_cols,
            ),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=int(cfg["training"]["n_estimators"]),
        random_state=int(cfg["training"]["random_state"]),
        n_jobs=-1,
    )
    return Pipeline([("preprocess", preprocess), ("model", model)])


def load_cleaned_data(cleaned_path: str | Path, cfg: dict[str, Any]) -> pd.DataFrame:
    """Load and validate M1's cleaned dataset. Never reads raw Craigslist CSV."""
    path = Path(cleaned_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Cleaned dataset not found at '{path}'. "
            "Run 'python -m src.data_workflow.pipeline' to generate it before training."
        )
    try:
        df = pd.read_csv(path)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError) as exc:
        raise RuntimeError(f"Could not read cleaned dataset '{path}': {exc}") from exc

    required = get_feature_columns(cfg) + [cfg["features"]["target"]]
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Cleaned dataset is missing required columns: {missing}")
    if len(df) < MIN_ROWS_REQUIRED:
        raise ValueError(
            f"Need at least {MIN_ROWS_REQUIRED} cleaned rows for train/test evaluation; found {len(df)}."
        )
    return df


def _evaluate_predictions(y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
    """Compute MAE (USD), RMSE (USD), and R2 on held-out test targets."""
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "r2": float(r2_score(y_true, y_pred)),
    }


def train_and_evaluate(df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[Pipeline, dict[str, Any]]:
    """Run one train/test split, fit baseline and Random Forest on train only, and score on test."""
    feature_cols = get_feature_columns(cfg)
    target_col = cfg["features"]["target"]
    required = feature_cols + [target_col]

    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Training DataFrame is missing required columns: {missing}")
    if len(df) < MIN_ROWS_REQUIRED:
        raise ValueError(
            f"Need at least {MIN_ROWS_REQUIRED} rows for train/test evaluation; found {len(df)}."
        )

    training_cfg = cfg["training"]
    random_state = int(training_cfg["random_state"])
    test_size = float(training_cfg["test_size"])
    max_rows = int(training_cfg["max_rows"])
    n_estimators = int(training_cfg["n_estimators"])

    source_cleaned_rows = int(len(df))
    sampled = source_cleaned_rows > max_rows
    work_df = (
        df.sample(n=max_rows, random_state=random_state)
        if sampled
        else df.copy()
    )

    X = work_df.loc[:, feature_cols]
    y = work_df.loc[:, target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
    )

    baseline = DummyRegressor(strategy="median").fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    baseline_scores = _evaluate_predictions(y_test, baseline_pred)

    pipeline = build_pipeline(cfg).fit(X_train, y_train)
    rf_pred = pipeline.predict(X_test)
    rf_scores = _evaluate_predictions(y_test, rf_pred)

    metrics: dict[str, Any] = {
        "rows": int(len(work_df)),
        "source_cleaned_rows": source_cleaned_rows,
        "sampled": sampled,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "mae": rf_scores["mae"],
        "rmse": rf_scores["rmse"],
        "r2": rf_scores["r2"],
        "baseline_mae": baseline_scores["mae"],
        "baseline_rmse": baseline_scores["rmse"],
        "baseline_r2": baseline_scores["r2"],
        "test_size": test_size,
        "random_state": random_state,
        "n_estimators": n_estimators,
        "max_rows": max_rows,
    }
    return pipeline, metrics


def save_artifacts(
    pipeline: Pipeline,
    metrics: dict[str, Any],
    model_path: str | Path,
    metrics_path: str | Path,
) -> None:
    """Persist fitted sklearn Pipeline and JSON evaluation summary."""
    model_dest = Path(model_path)
    model_dest.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_dest)

    metric_dest = Path(metrics_path)
    metric_dest.parent.mkdir(parents=True, exist_ok=True)
    metric_dest.write_text(json.dumps(metrics, indent=2), encoding="utf-8")


def run_training(cfg: dict[str, Any] | None = None, config_path: str | Path = "config.yaml") -> dict[str, Any]:
    """Execute end-to-end training from M1's cleaned CSV and export artifacts."""
    active_cfg = load_config(config_path) if cfg is None else cfg
    paths = active_cfg["paths"]

    df = load_cleaned_data(paths["cleaned_data"], active_cfg)
    LOG.info("Loaded %d cleaned rows from %s", len(df), paths["cleaned_data"])

    pipeline, metrics = train_and_evaluate(df, active_cfg)
    save_artifacts(pipeline, metrics, paths["model"], paths["metrics"])
    LOG.info("Saved model to %s and metrics to %s", paths["model"], paths["metrics"])
    LOG.info("Evaluation metrics: %s", metrics)
    return metrics


def main() -> None:
    """CLI entry point: python -m src.modeling.train."""
    setup_logging()
    run_training()


if __name__ == "__main__":
    main()
