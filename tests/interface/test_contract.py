"""M3-owned inference schema contract check without requiring Streamlit UI session."""
from src.modeling.inference import validate_car

def test_form_columns_match_model_schema():
    cfg={"cleaning":{"min_year":1980,"max_year":2027,"max_odometer":500000}}
    frame=validate_car(2018,55000,"Toyota","gas","automatic",cfg)
    assert list(frame.columns)==["year","odometer","manufacturer","fuel","transmission"]
