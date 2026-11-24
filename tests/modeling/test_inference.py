"""Unit and integration tests for M2 inference validation and saved-model loading."""

from __future__ import annotations

import math

import joblib
import pandas as pd
import pytest

from src.modeling.inference import FEATURE_COLUMNS, load_model, predict_price, validate_car
from src.modeling.train import build_pipeline

CFG = {
    "cleaning": {
        "min_year": 1980,
        "max_year": 2027,
        "max_odometer": 500000,
    },
    "features": {
        "numeric": ["year", "odometer"],
        "categorical": ["manufacturer", "fuel", "transmission"],
        "target": "price",
    },
    "training": {
        "n_estimators": 8,
        "random_state": 42,
    },
}


def _train_and_save_model(tmp_path):
    rows = pd.DataFrame(
        [
            {
                "year": 2012 + (i % 10),
                "odometer": 20000.0 + i * 5000.0,
                "manufacturer": "toyota" if i % 2 == 0 else "ford",
                "fuel": "gas" if i % 3 else "hybrid",
                "transmission": "automatic" if i % 2 == 0 else "manual",
            }
            for i in range(20)
        ]
    )
    y = [12000.0 + (i % 10) * 1200.0 - i * 100.0 for i in range(20)]
    pipeline = build_pipeline(CFG).fit(rows, y)
    model_path = tmp_path / "price_pipeline.joblib"
    joblib.dump(pipeline, model_path)
    return model_path


def test_valid_car():
    frame = validate_car(2018, 60000, "  Toyota ", " GAS", "Automatic ", CFG)
    assert list(frame.columns) == FEATURE_COLUMNS
    assert frame.iloc[0]["year"] == 2018
    assert frame.iloc[0]["odometer"] == 60000.0
    assert frame.iloc[0]["manufacturer"] == "toyota"
    assert frame.iloc[0]["fuel"] == "gas"
    assert frame.iloc[0]["transmission"] == "automatic"


def test_negative_odometer_rejected():
    with pytest.raises(ValueError, match="Odometer"):
        validate_car(2018, -1, "toyota", "gas", "automatic", CFG)


def test_excessive_odometer_rejected():
    with pytest.raises(ValueError, match="Odometer"):
        validate_car(2018, 500001, "toyota", "gas", "automatic", CFG)


def test_non_finite_odometer_rejected():
    with pytest.raises(ValueError, match="Odometer"):
        validate_car(2018, float("nan"), "toyota", "gas", "automatic", CFG)


def test_invalid_year_rejected():
    with pytest.raises(ValueError, match="Year"):
        validate_car(1970, 50000, "toyota", "gas", "automatic", CFG)
    with pytest.raises(ValueError, match="Year"):
        validate_car(2035, 50000, "toyota", "gas", "automatic", CFG)


def test_boolean_numeric_inputs_rejected():
    with pytest.raises(ValueError, match="Year"):
        validate_car(True, 50000, "toyota", "gas", "automatic", CFG)
    with pytest.raises(ValueError, match="Odometer"):
        validate_car(2018, False, "toyota", "gas", "automatic", CFG)


def test_blank_categorical_rejected():
    with pytest.raises(ValueError, match="Manufacturer, fuel and transmission"):
        validate_car(2018, 50000, "   ", "gas", "automatic", CFG)


def test_saved_pipeline_reloads_and_predicts_valid_usd(tmp_path):
    model_path = _train_and_save_model(tmp_path)
    car = validate_car(2018, 55000, "Toyota", "gas", "automatic", CFG)

    loaded = load_model(model_path)
    assert loaded is not None

    price = predict_price(model_path, car)
    assert isinstance(price, float)
    assert math.isfinite(price)
    assert price > 0.0


def test_unseen_manufacturer_does_not_crash_prediction(tmp_path):
    model_path = _train_and_save_model(tmp_path)

    # Manufacturer "rivian" and "unknown_brand_xyz" were never present in training data
    for unseen_make, unseen_fuel, unseen_trans in [
        ("Rivian", "electric", "automatic"),
        ("unknown_brand_xyz", "hydrogen", "cvt"),
    ]:
        car = validate_car(2021, 25000, unseen_make, unseen_fuel, unseen_trans, CFG)
        price = predict_price(model_path, car)
        assert isinstance(price, float)
        assert math.isfinite(price)
        assert price >= 0.0


def test_missing_model_artifact_raises_clear_error(tmp_path):
    missing_path = tmp_path / "does_not_exist.joblib"
    car = validate_car(2018, 55000, "toyota", "gas", "automatic", CFG)

    with pytest.raises(FileNotFoundError, match=r"python -m src\.modeling\.train"):
        predict_price(missing_path, car)


def test_predict_price_rejects_missing_columns_and_empty_frame(tmp_path):
    model_path = _train_and_save_model(tmp_path)

    incomplete = pd.DataFrame([{"year": 2018, "odometer": 50000.0, "manufacturer": "toyota"}])
    with pytest.raises(ValueError, match="missing required columns"):
        predict_price(model_path, incomplete)

    empty = pd.DataFrame(columns=FEATURE_COLUMNS)
    with pytest.raises(ValueError, match="non-empty pandas DataFrame"):
        predict_price(model_path, empty)
