# M2: Model, Metrics, and Limitations

*Author: Ibrahim Alhajjaj (M2 — Machine Learning Engineering)*  
*Note for M4 integration: Replace `[PENDING]` placeholders with exact values from `outputs/metrics/metrics.json` after M1 supplies `data/processed/vehicles_clean.csv` and `python -m src.modeling.train` is executed.*

---

## Slide 1: Why Machine Learning for Used-Car Pricing?

- **Problem type:** Supervised tabular regression predicting Craigslist listing price (`price`, USD).
- **Why ML beats fixed rules or averages:**
  - Car depreciation is **non-linear** across model `year` and `odometer` (miles).
  - Price depends on **feature interactions** (e.g., mileage penalty differs across `manufacturer`, `fuel`, and `transmission`).
- **Model choice:** `RandomForestRegressor` (`n_estimators = 120`, `random_state = 42`) captures non-linearities and interactions without manual feature engineering.

---

## Slide 2: Leakage-Safe Modeling Pipeline

- **Input contract (from M1's `data/processed/vehicles_clean.csv` only):**
  - Numeric: `year`, `odometer` (miles)
  - Categorical: `manufacturer`, `fuel`, `transmission`
  - Target: `price` (USD asking price)
- **Single deterministic hold-out split:** `80%` train / `20%` test (`random_state = 42`), with `max_rows = 100,000` deterministic sample cap for local reproducibility.
- **Zero data leakage:**
  - `SimpleImputer` (median for numeric, most-frequent for categorical) and `OneHotEncoder(handle_unknown="ignore")` live inside a scikit-learn `Pipeline` fitted **strictly on `X_train`**.
  - Unseen manufacturers at inference time never crash M3's Streamlit app.

---

## Slide 3: Baseline vs. Random Forest Evaluation (Held-Out Test Set)

- **Fair comparison:** Median `DummyRegressor` and `RandomForestRegressor` are fitted on the exact same training split and evaluated on the **exact same held-out test rows**.
- **Dataset counts (`outputs/metrics/metrics.json`):**
  - Modeled rows: `[PENDING: run training after M1 generates data/processed/vehicles_clean.csv]`
  - Train / Test split: `[PENDING: run training after M1 generates data/processed/vehicles_clean.csv]`

| Model | Test MAE (USD) | Test RMSE (USD) | Test $R^2$ |
|---|---|---|---|
| Median Baseline (`DummyRegressor`) | [PENDING: run training after M1 generates data/processed/vehicles_clean.csv] | [PENDING: run training after M1 generates data/processed/vehicles_clean.csv] | [PENDING: run training after M1 generates data/processed/vehicles_clean.csv] |
| Random Forest (`120` trees) | [PENDING: run training after M1 generates data/processed/vehicles_clean.csv] | [PENDING: run training after M1 generates data/processed/vehicles_clean.csv] | [PENDING: run training after M1 generates data/processed/vehicles_clean.csv] |

---

## Slide 4: Error Behavior & Honest Limitations

- **Asking vs. sale price:** Craigslist listings reflect advertised asking prices, not verified final sale prices.
- **Omitted vehicle details:** Specific model/trim, condition, salvage/clean title, drivetrain, and location are not in the 5-feature contract, creating natural within-brand price spread.
- **Dataset & market bias:** Historical regional snapshot; prices drift over time and may include noisy or duplicate listings.
- **Model constraints:**
  - `max_rows = 100,000` sampling cap trades full-dataset coverage for reproducible local compute.
  - Random Forests cannot extrapolate beyond observed training price ranges.
- **Safe engineering contract:** Saved artifact (`outputs/models/price_pipeline.joblib`) verified by automated `pytest` tests for reloadability, leakage prevention, unseen categories, and missing-model error handling.
