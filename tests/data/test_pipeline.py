"""Data preparation checks."""
import pandas as pd
import pytest
from src.data_workflow.pipeline import validate, clean

CFG = {'features': {'numeric': ['year','odometer'], 'categorical':['manufacturer','fuel','transmission'], 'target':'price'}, 'cleaning':{'min_price':500,'max_price':150000,'min_year':1980,'max_year':2027,'max_odometer':500000}}

def test_missing_columns():
    with pytest.raises(ValueError, match='missing columns'):
        validate(pd.DataFrame({'price':[1000]}), CFG)

def test_clean_filters_invalid_prices():
    data = pd.DataFrame({'year':[2018,2019], 'odometer':[10000,20000], 'manufacturer':['toyota','ford'], 'fuel':['gas','gas'], 'transmission':['automatic','manual'], 'price':[10000,-5]})
    assert len(clean(validate(data, CFG), CFG)) == 1
