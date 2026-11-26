"""Unit and integration tests for the M2 training pipeline."""

from __future__ import annotations

import json
import math

import joblib
import numpy as np
import pandas as pd
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from src.modeling.inference import predict_price
from src.modeling.train import (
    build_pipeline,
    run_training,
    train_and_evaluate,
)


def _make_cfg(tmp_path=None, *, max_rows: int = 1000, n_estimators: int = 10) -> dict:
    paths = {}
    if tmp_path is not None:
        paths = {
            "cleaned_data": str(tmp_path / "vehicles_clean.csv"),
            "model": str(tmp_path / "outputs" / "models" / "price_pipeline.joblib"),
            "metrics": str(tmp_path / "outputs" / "metrics" / "metrics.json"),
        }
    return {
        "paths": paths,
        "features": {
            "numeric": ["year", "odometer"],
            "categorical": ["manufacturer", "fuel", "transmission"],
            "target": "price",
        },
        "training": {
            "test_size": 0.2,
            "random_state": 42,
            "n_estimators": n_estimators,
            "max_rows": max_rows,
        },
    }


def _make_synthetic_dataset(n_rows: int = 40) -> pd.DataFrame:
    manufacturers = ["toyota", "ford", "honda", "bmw"]
    fuels = ["gas", "diesel", "hybrid"]
    transmissions = ["automatic", "manual"]
    rows = []
    for i in range(n_rows):
        year = 2005 + (i % 18)
        odometer = 15000 + (i * 4200)
        make = manufacturers[i % len(manufacturers)]
        fuel = fuels[i % len(fuels)]
        trans = transmissions[i % len(transmissions)]
        brand_bonus = {"bmw": 6500, "toyota": 2500, "honda": 2000, "ford": 1000}[make]
        price = max(
            1500.0,
            4000.0 + (year - 2000) * 1100.0 - odometer * 0.04 + brand_bonus,
        )
        rows.append(
            {
                "year": year,
                "odometer": float(odometer),
                "manufacturer": make,
                "fuel": fuel,
                "transmission": trans,
                "price": float(price),
            }
        )
    return pd.DataFrame(rows)


def test_small_training_pipeline():
    cfg = _make_cfg(n_estimators=5)
    rows = pd.DataFrame(
        [
            {
                "year": 2015 + (i % 6),
                "odometer": 10000 + i * 800,
                "manufacturer": "toyota" if i % 2 else "ford",
                "fuel": "gas",
                "transmission": "automatic",
            }
            for i in range(12)
        ]
    )
    y = [10000 + i * 500 for i in range(12)]
    model = build_pipeline(cfg).fit(rows, y)
    preds = model.predict(rows.iloc[:2])
    assert len(preds) == 2
    assert all(math.isfinite(float(p)) for p in preds)


def test_post_split_fitting_prevents_leakage():
    cfg = _make_cfg(n_estimators=10)
    df = _make_synthetic_dataset(n_rows=30)

    # Identify exact test indices for this split so we can inject test-only values
    # and prove neither imputer nor OneHotEncoder sees test rows during fit().
    _, test_idx = train_test_split(
        df.index,
        test_size=cfg["training"]["test_size"],
        random_state=cfg["training"]["random_state"],
    )
    train_idx = df.index.difference(test_idx)

    # Make test-only manufacturer and extreme test-only odometer values
    df.loc[test_idx, "manufacturer"] = "test_only_manufacturer"
    df.loc[test_idx, "odometer"] = 499999.0

    # Verify full-dataset median genuinely differs from train-only median
    train_median_odometer = float(df.loc[train_idx, "odometer"].median())
    full_median_odometer = float(df["odometer"].median())
    assert train_median_odometer != full_median_odometer

    pipeline, _ = train_and_evaluate(df, cfg)

    preprocess = pipeline.named_steps["preprocess"]
    num_imputer = preprocess.named_transformers_["num"].named_steps["impute"]
    cat_encoder = preprocess.named_transformers_["cat"].named_steps["encode"]

    # Numeric imputer statistics must come strictly from X_train
    assert num_imputer.statistics_[1] == pytest.approx(train_median_odometer)
    assert num_imputer.statistics_[1] != pytest.approx(full_median_odometer)

    # OneHotEncoder must never learn categories that exist only in X_test
    learned_manufacturers = set(cat_encoder.categories_[0])
    assert "test_only_manufacturer" not in learned_manufacturers

    # Yet predicting on X_test with the unseen test-only manufacturer must succeed
    test_features = df.loc[test_idx, ["year", "odometer", "manufacturer", "fuel", "transmission"]]
    preds = pipeline.predict(test_features)
    assert len(preds) == len(test_idx)
    assert all(math.isfinite(float(p)) for p in preds)


def test_baseline_and_rf_evaluated_on_identical_test_rows():
    cfg = _make_cfg(n_estimators=20)
    df = _make_synthetic_dataset(n_rows=50)

    pipeline, metrics = train_and_evaluate(df, cfg)

    feature_cols = ["year", "odometer", "manufacturer", "fuel", "transmission"]
    X_train, X_test, y_train, y_test = train_test_split(
        df[feature_cols],
        df["price"],
        test_size=cfg["training"]["test_size"],
        random_state=cfg["training"]["random_state"],
    )

    expected_baseline = DummyRegressor(strategy="median").fit(X_train, y_train)
    expected_baseline_pred = expected_baseline.predict(X_test)
    expected_rf_pred = pipeline.predict(X_test)

    assert metrics["rows"] == 50
    assert metrics["train_rows"] == len(X_train)
    assert metrics["test_rows"] == len(X_test)
    assert metrics["train_rows"] + metrics["test_rows"] == metrics["rows"]

    assert metrics["baseline_mae"] == pytest.approx(
        mean_absolute_error(y_test, expected_baseline_pred)
    )
    assert metrics["baseline_rmse"] == pytest.approx(
        float(np.sqrt(mean_squared_error(y_test, expected_baseline_pred)))
    )
    assert metrics["baseline_r2"] == pytest.approx(
        r2_score(y_test, expected_baseline_pred)
    )
    assert metrics["mae"] == pytest.approx(mean_absolute_error(y_test, expected_rf_pred))
    assert metrics["rmse"] == pytest.approx(
        float(np.sqrt(mean_squared_error(y_test, expected_rf_pred)))
    )
    assert metrics["r2"] == pytest.approx(r2_score(y_test, expected_rf_pred))

    # On structured synthetic data, Random Forest should beat the constant median baseline
    assert metrics["mae"] < metrics["baseline_mae"]
    assert metrics["r2"] > metrics["baseline_r2"]


def test_run_training_writes_joblib_and_metrics_json(tmp_path):
    cfg = _make_cfg(tmp_path=tmp_path, n_estimators=10)
    df = _make_synthetic_dataset(n_rows=40)
    df.to_csv(cfg["paths"]["cleaned_data"], index=False)

    returned_metrics = run_training(cfg=cfg)

    model_file = tmp_path / "outputs" / "models" / "price_pipeline.joblib"
    metrics_file = tmp_path / "outputs" / "metrics" / "metrics.json"
    assert model_file.is_file()
    assert metrics_file.is_file()

    saved_metrics = json.loads(metrics_file.read_text(encoding="utf-8"))
    assert saved_metrics == returned_metrics

    required_keys = {
        "rows",
        "source_cleaned_rows",
        "sampled",
        "train_rows",
        "test_rows",
        "mae",
        "rmse",
        "r2",
        "baseline_mae",
        "baseline_rmse",
        "baseline_r2",
    }
    assert required_keys.issubset(saved_metrics.keys())
    assert saved_metrics["rows"] == 40
    assert saved_metrics["source_cleaned_rows"] == 40
    assert saved_metrics["sampled"] is False
    assert saved_metrics["train_rows"] == 32
    assert saved_metrics["test_rows"] == 8
    for metric_key in ("mae", "rmse", "r2", "baseline_mae", "baseline_rmse", "baseline_r2"):
        assert math.isfinite(saved_metrics[metric_key])

    # Reload saved pipeline directly and through predict_price
    reloaded = joblib.load(model_file)
    sample = pd.DataFrame(
        [
            {
                "year": 2018,
                "odometer": 55000.0,
                "manufacturer": "toyota",
                "fuel": "gas",
                "transmission": "automatic",
            }
        ]
    )
    direct_pred = float(reloaded.predict(sample)[0])
    api_pred = predict_price(model_file, sample)
    assert math.isfinite(direct_pred)
    assert api_pred == pytest.approx(max(0.0, direct_pred))
    assert api_pred > 0.0


def test_sample_cap_enforced_deterministically():
    cfg = _make_cfg(max_rows=25, n_estimators=8)
    df = _make_synthetic_dataset(n_rows=40)

    _, metrics_first = train_and_evaluate(df, cfg)
    _, metrics_second = train_and_evaluate(df, cfg)

    assert metrics_first["source_cleaned_rows"] == 40
    assert metrics_first["rows"] == 25
    assert metrics_first["sampled"] is True
    assert metrics_first["train_rows"] + metrics_first["test_rows"] == 25
    assert metrics_first == metrics_second


def test_missing_cleaned_csv_raises_clear_error(tmp_path):
    cfg = _make_cfg(tmp_path=tmp_path)
    with pytest.raises(FileNotFoundError, match=r"src\.data_workflow\.pipeline"):
        run_training(cfg=cfg)


def test_missing_required_columns_raises_value_error():
    cfg = _make_cfg()
    bad_df = _make_synthetic_dataset(n_rows=15).drop(columns=["transmission"])
    with pytest.raises(ValueError, match="missing required columns"):
        train_and_evaluate(bad_df, cfg)


def test_too_few_rows_raises_value_error():
    cfg = _make_cfg()
    tiny_df = _make_synthetic_dataset(n_rows=5)
    with pytest.raises(ValueError, match="at least 10"):
        train_and_evaluate(tiny_df, cfg)
