"""Train, evaluate and save a leakage-safe regression pipeline."""
import json
import logging
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.dummy import DummyRegressor
from src.shared.config import load_config, setup_logging

LOG = logging.getLogger(__name__)

def build_pipeline(cfg):
    """Build fit-on-train-only preprocessing and regression pipeline."""
    num = cfg['features']['numeric']; cat = cfg['features']['categorical']
    preprocess = ColumnTransformer([
        ('num', Pipeline([('impute', SimpleImputer(strategy='median'))]), num),
        ('cat', Pipeline([('impute', SimpleImputer(strategy='most_frequent')), ('encode', OneHotEncoder(handle_unknown='ignore'))]), cat),
    ])
    model = RandomForestRegressor(n_estimators=cfg['training']['n_estimators'], random_state=cfg['training']['random_state'], n_jobs=-1)
    return Pipeline([('preprocess', preprocess), ('model', model)])

def main():
    """Train/test split, baseline comparison, evaluation and model export."""
    setup_logging(); cfg = load_config(); p = cfg['paths']; t = cfg['training']
    try:
        df = pd.read_csv(p['cleaned_data'])
    except OSError as exc:
        raise RuntimeError('Run python -m src.data_pipeline before training') from exc
    if len(df) < 10:
        raise ValueError('Need at least 10 cleaned rows for train/test evaluation')
    if len(df) > t['max_rows']:
        df = df.sample(n=t['max_rows'], random_state=t['random_state'])
    cols = cfg['features']['numeric'] + cfg['features']['categorical']
    X, y = df[cols], df[cfg['features']['target']]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=t['test_size'], random_state=t['random_state'])
    baseline = DummyRegressor(strategy='median').fit(X_train, y_train)
    baseline_pred = baseline.predict(X_test)
    pipeline = build_pipeline(cfg).fit(X_train, y_train)
    pred = pipeline.predict(X_test)
    metrics = {'rows': int(len(df)), 'test_rows': int(len(X_test)), 'mae': float(mean_absolute_error(y_test, pred)), 'rmse': float(np.sqrt(mean_squared_error(y_test, pred))), 'r2': float(r2_score(y_test, pred)), 'baseline_mae': float(mean_absolute_error(y_test, baseline_pred)), 'baseline_rmse': float(np.sqrt(mean_squared_error(y_test, baseline_pred)))}
    model_path = Path(p['model']); model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, model_path)
    metric_path = Path(p['metrics']); metric_path.parent.mkdir(parents=True, exist_ok=True)
    metric_path.write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    LOG.info('Saved model to %s; metrics=%s', model_path, metrics)

if __name__ == '__main__':
    main()
