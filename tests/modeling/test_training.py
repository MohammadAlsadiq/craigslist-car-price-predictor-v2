"""Verify a fitted sklearn pipeline accepts the agreed feature schema."""
import pandas as pd
from src.modeling.train import build_pipeline

def test_small_training_pipeline():
    cfg = {"features":{"numeric":["year","odometer"],"categorical":["manufacturer","fuel","transmission"]},"training":{"n_estimators":3,"random_state":42}}
    rows = pd.DataFrame([
        {"year":2015+i,"odometer":10000+i*800,"manufacturer":"toyota" if i%2 else "ford","fuel":"gas","transmission":"automatic"} for i in range(12)
    ])
    y = [10000+i*500 for i in range(12)]
    model = build_pipeline(cfg).fit(rows,y)
    assert len(model.predict(rows.iloc[:2])) == 2
