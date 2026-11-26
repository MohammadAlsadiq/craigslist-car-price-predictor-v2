
"""M3-owned inference schema contract checks."""

import pytest
from src.modeling.inference import validate_car


def test_form_columns_match_model_schema():
    cfg = {
        "cleaning": {
            "min_year": 1980,
            "max_year": 2027,
            "max_odometer": 500000
        }
    }

    frame = validate_car(
        2018, 55000, "Toyota", "gas", "automatic", cfg
    )

    assert list(frame.columns) == [
        "year",
        "odometer",
        "manufacturer",
        "fuel",
        "transmission"
    ]


def test_negative_odometer_is_rejected():
    cfg = {
        "cleaning": {
            "min_year": 1980,
            "max_year": 2027,
            "max_odometer": 500000
        }
    }

    with pytest.raises(ValueError):
        validate_car(
            2018, -1, "Toyota", "gas", "automatic", cfg
        )


def test_empty_manufacturer_is_rejected():
    cfg = {
        "cleaning": {
            "min_year": 1980,
            "max_year": 2027,
            "max_odometer": 500000
        }
    }

    with pytest.raises(ValueError):
        validate_car(
            2018, 55000, "", "gas", "automatic", cfg
        )


def test_year_outside_limits_is_rejected():
    cfg = {
        "cleaning": {
            "min_year": 1980,
            "max_year": 2027,
            "max_odometer": 500000
        }
    }

    with pytest.raises(ValueError):
        validate_car(
            1970, 55000, "Toyota", "gas", "automatic", cfg
        )
