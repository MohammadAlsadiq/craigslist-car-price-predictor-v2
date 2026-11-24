"""Input validation and saved-model inference for used-car price prediction.

Provides the documented M2 -> M3 contract used by ``src/interface/ui.py``:
- ``validate_car(year, odometer, manufacturer, fuel, transmission, cfg)``
- ``load_model(model_path)``
- ``predict_price(model_path, frame)``
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import joblib
import pandas as pd

FEATURE_COLUMNS = ["year", "odometer", "manufacturer", "fuel", "transmission"]
CATEGORICAL_COLUMNS = ["manufacturer", "fuel", "transmission"]


def validate_car(
    year: int,
    odometer: float,
    manufacturer: str,
    fuel: str,
    transmission: str,
    cfg: dict[str, Any],
) -> pd.DataFrame:
    """Validate and normalize the 5 agreed car attributes into a 1-row DataFrame."""
    cleaning = cfg["cleaning"]

    if isinstance(year, bool) or not isinstance(year, (int, float)) or not math.isfinite(year):
        raise ValueError("Year must be a finite numeric value")
    if int(year) != year:
        raise ValueError("Year must be a whole number")
    year_int = int(year)
    if not cleaning["min_year"] <= year_int <= cleaning["max_year"]:
        raise ValueError(
            f"Year must be between {cleaning['min_year']} and {cleaning['max_year']}"
        )

    if isinstance(odometer, bool) or not isinstance(odometer, (int, float)) or not math.isfinite(odometer):
        raise ValueError("Odometer must be a finite numeric value")
    odometer_val = float(odometer)
    if not 0 <= odometer_val <= cleaning["max_odometer"]:
        raise ValueError(
            f"Odometer must be between 0 and {cleaning['max_odometer']:,}"
        )

    if not all(isinstance(v, str) and v.strip() for v in (manufacturer, fuel, transmission)):
        raise ValueError("Manufacturer, fuel and transmission are required")

    return pd.DataFrame(
        [
            {
                "year": year_int,
                "odometer": odometer_val,
                "manufacturer": manufacturer.strip().lower(),
                "fuel": fuel.strip().lower(),
                "transmission": transmission.strip().lower(),
            }
        ],
        columns=FEATURE_COLUMNS,
    )


def load_model(model_path: str | Path) -> Any:
    """Load a trusted local joblib pipeline artifact or raise a helpful FileNotFoundError."""
    path = Path(model_path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Model artifact not found at '{path}'. "
            "Run 'python -m src.modeling.train' after generating data/processed/vehicles_clean.csv."
        )
    # Only load artifacts created by this trusted repository.
    return joblib.load(path)


def _prepare_inference_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Verify required columns exist and normalize categorical strings defensively."""
    if not isinstance(frame, pd.DataFrame) or frame.empty:
        raise ValueError("Inference input must be a non-empty pandas DataFrame")

    missing = [col for col in FEATURE_COLUMNS if col not in frame.columns]
    if missing:
        raise ValueError(f"Inference DataFrame is missing required columns: {missing}")

    prepared = frame.loc[:, FEATURE_COLUMNS].copy()
    for col in CATEGORICAL_COLUMNS:
        prepared[col] = (
            prepared[col]
            .fillna("unknown")
            .astype(str)
            .str.strip()
            .str.lower()
            .replace("", "unknown")
        )
    return prepared


def predict_price(model_path: str | Path, frame: pd.DataFrame) -> float:
    """Load the saved pipeline and return a finite non-negative predicted price in USD."""
    prepared = _prepare_inference_frame(frame)
    model = load_model(model_path)
    raw_prediction = float(model.predict(prepared)[0])
    if not math.isfinite(raw_prediction):
        raise ValueError("Model produced a non-finite price prediction")
    return max(0.0, raw_prediction)
