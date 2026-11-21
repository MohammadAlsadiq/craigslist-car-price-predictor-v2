"""Unit tests for inference validation."""
import pytest
from src.modeling.inference import validate_car

CFG = {'cleaning': {'min_year': 1980, 'max_year': 2027, 'max_odometer': 500000}}

def test_valid_car():
    frame = validate_car(2018, 60000, 'Toyota', 'gas', 'automatic', CFG)
    assert frame.iloc[0]['manufacturer'] == 'toyota'

def test_negative_odometer_rejected():
    with pytest.raises(ValueError, match='Odometer'):
        validate_car(2018, -1, 'toyota', 'gas', 'automatic', CFG)

def test_invalid_year_rejected():
    with pytest.raises(ValueError, match='Year'):
        validate_car(1970, 50000, 'toyota', 'gas', 'automatic', CFG)
