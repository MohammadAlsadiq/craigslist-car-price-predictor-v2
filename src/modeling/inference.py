"""Input validation and saved-model inference."""
from pathlib import Path
import joblib
import pandas as pd

def validate_car(year, odometer, manufacturer, fuel, transmission, cfg):
    """Reject invalid car specifications before inference."""
    c = cfg['cleaning']
    if not c['min_year'] <= year <= c['max_year']:
        raise ValueError(f'Year must be between {c["min_year"]} and {c["max_year"]}')
    if not 0 <= odometer <= c['max_odometer']:
        raise ValueError(f'Odometer must be between 0 and {c["max_odometer"]:,}')
    if not all(isinstance(v, str) and v.strip() for v in (manufacturer, fuel, transmission)):
        raise ValueError('Manufacturer, fuel and transmission are required')
    return pd.DataFrame([{'year': year, 'odometer': odometer, 'manufacturer': manufacturer.strip().lower(), 'fuel': fuel.strip().lower(), 'transmission': transmission.strip().lower()}])

def predict_price(model_path, frame):
    """Load a trusted local joblib artifact and return predicted USD price."""
    if not Path(model_path).is_file():
        raise FileNotFoundError('Model not trained yet; run python -m src.train')
    model = joblib.load(model_path)  # Only load artifacts created by this trusted repository.
    return max(0.0, float(model.predict(frame)[0]))
