## AI Method and Evaluation (M2)

*Author: Ibrahim Alhajjaj (M2 — Machine Learning Engineering)*

### 1. Why Machine Learning Is Appropriate for This Problem

Estimating used-car listing prices from vehicle specifications is a supervised tabular regression problem that is poorly served by static lookup tables or simple linear depreciation rules:

- **Non-linear depreciation:** Vehicle value drops steeply in the first few years after manufacture and flattens for older vehicles, while high mileage reduces value at different rates depending on vehicle age and powertrain.
- **Feature interactions:** The price impact of mileage or model year depends strongly on categorical attributes such as `manufacturer` (e.g., luxury vs. economy brands), `fuel` (e.g., diesel trucks or hybrids vs. standard gasoline sedans), and `transmission`.
- **High-cardinality heterogeneity:** Craigslist listings span dozens of manufacturers and thousands of year-mileage combinations. A non-parametric ensemble model (`RandomForestRegressor`) learns non-linear thresholds and multi-way feature interactions directly from historical listings without requiring manual interaction engineering.

### 2. Agreed Feature Schema and Target

The modeling pipeline (`src/modeling/train.py`) reads configuration directly from `config.yaml` and consumes **only** M1's cleaned dataset (`data/processed/vehicles_clean.csv`). Raw Craigslist data (`data/raw/vehicles.csv`) is never accessed during training.

| Column | Role | Type & Units | Description |
|---|---|---|---|
| `year` | Numeric feature | Integer | Vehicle model year (cleaned range: 1980–2027) |
| `odometer` | Numeric feature | Float (miles) | Reported mileage in miles (cleaned range: 0–500,000) |
| `manufacturer` | Categorical feature | Lowercase string | Vehicle brand/make |
| `fuel` | Categorical feature | Lowercase string | Fuel type (`gas`, `diesel`, `hybrid`, `electric`, `other`, etc.) |
| `transmission` | Categorical feature | Lowercase string | Transmission type (`automatic`, `manual`, `other`, etc.) |
| `price` | Target variable | Float (USD) | Advertised Craigslist listing price in USD (cleaned range: \$500–\$150,000) |

### 3. Train/Test Split, Sampling Cap, and Leakage Safeguards

To ensure an honest and reproducible out-of-sample evaluation, `src/modeling/train.py` enforces the following experimental controls:

1. **Deterministic Sampling Cap:** Because the full cleaned Kaggle Craigslist dataset can contain hundreds of thousands of rows, `config.yaml` specifies `max_rows: 100000`. If `vehicles_clean.csv` exceeds `100,000` rows, the script draws a reproducible random sample (`random_state = 42`) of `100,000` rows prior to splitting and explicitly logs both `source_cleaned_rows` and the modeled `rows` count in `outputs/metrics/metrics.json`.
2. **Single Hold-Out Split:** Data is partitioned once via `train_test_split` with `test_size = 0.2` (80% training, 20% held-out test) and `random_state = 42`. All reported metrics are computed exclusively on this held-out test split; validation scores are never substituted for test scores.
3. **Post-Split Preprocessing (Zero Data Leakage):** All data-dependent transformations live inside a single scikit-learn `Pipeline` fitted strictly on `(X_train, y_train)`:
   - Numeric columns (`year`, `odometer`): `SimpleImputer(strategy="median")`
   - Categorical columns (`manufacturer`, `fuel`, `transmission`): `SimpleImputer(strategy="most_frequent")` followed by `OneHotEncoder(handle_unknown="ignore")`
   Fitting inside the pipeline after splitting guarantees that numeric medians, categorical modes, and one-hot category vocabularies never inspect test rows. Setting `handle_unknown="ignore"` also ensures unseen manufacturers or rare categories entered in M3's Streamlit UI never crash inference.

### 4. Baseline vs. Random Forest Experimental Design

To prove that the machine learning model adds measurable value over a naive heuristic, two models are fitted on the **exact same training split** and scored on the **exact same held-out test rows**:

- **Baseline (`DummyRegressor(strategy="median")`):** Predicts the training-set median listing price for every vehicle in the test set.
- **Primary Model (`RandomForestRegressor`):** Configured via `config.yaml` with `n_estimators = 120`, `random_state = 42`, and `n_jobs = -1`, wrapped with the leakage-safe preprocessing `ColumnTransformer` and serialized to `outputs/models/price_pipeline.joblib`.

Both models are evaluated using three complementary metrics:
- **MAE (Mean Absolute Error, USD):** Average absolute dollar gap between predicted and listed prices; robust to extreme outliers and directly interpretable to car buyers/sellers.
- **RMSE (Root Mean Squared Error, USD):** Square root of average squared dollar error; penalizes large misses more heavily than MAE and highlights tail-error behavior.
- **$R^2$ (Coefficient of Determination):** Proportion of test-set price variance explained by the model ($R^2 \le 0$ indicates performance no better than predicting a constant).

### 5. Held-Out Test Evaluation Results

> **Status:** `data/processed/vehicles_clean.csv` has not yet been generated by M1 in this workspace. Per project integrity rules, no metrics or row counts are fabricated or estimated. Populate the tables below directly from `outputs/metrics/metrics.json` immediately after running `python -m src.modeling.train`.

#### Dataset and Split Row Counts

| Field (`outputs/metrics/metrics.json`) | Value |
|---|---|
| Cleaned rows before cap (`source_cleaned_rows`) | [PENDING: run training after M1 generates data/processed/vehicles_clean.csv] |
| Modeled rows (`rows`) | [PENDING: run training after M1 generates data/processed/vehicles_clean.csv] |
| Sample cap applied (`sampled`, `max_rows = 100000`) | [PENDING: run training after M1 generates data/processed/vehicles_clean.csv] |
| Training rows (`train_rows`, 80%) | [PENDING: run training after M1 generates data/processed/vehicles_clean.csv] |
| Held-out test rows (`test_rows`, 20%) | [PENDING: run training after M1 generates data/processed/vehicles_clean.csv] |

#### Held-Out Test Metrics Comparison

| Model | Test MAE (USD) | Test RMSE (USD) | Test $R^2$ |
|---|---|---|---|
| Median Baseline (`DummyRegressor`) | [PENDING: run training after M1 generates data/processed/vehicles_clean.csv] | [PENDING: run training after M1 generates data/processed/vehicles_clean.csv] | [PENDING: run training after M1 generates data/processed/vehicles_clean.csv] |
| Random Forest (`RandomForestRegressor`, 120 trees) | [PENDING: run training after M1 generates data/processed/vehicles_clean.csv] | [PENDING: run training after M1 generates data/processed/vehicles_clean.csv] | [PENDING: run training after M1 generates data/processed/vehicles_clean.csv] |

### 6. Error Analysis

Once real test predictions are produced on `data/processed/vehicles_clean.csv`, error interpretation focuses on three diagnostic checks:

- **MAE vs. RMSE gap:** A substantially larger RMSE than MAE indicates right-skewed residual errors—typically caused by luxury, collector, specialty truck, or mispriced listings where dollar residuals are much larger than for mainstream sedans.
- **Within-make model/trim dispersion:** Because the agreed schema includes `manufacturer` (e.g., `ford`) without specific model or trim (e.g., Fiesta vs. F-350 Super Duty), two vehicles with identical `year`, `odometer`, `fuel`, and `transmission` can legitimately differ by tens of thousands of dollars, creating an irreducible noise floor for any model using these 5 features alone.
- **Rare and unseen categories:** Listings with rare or unseen manufacturers fall back to an all-zero one-hot vector under `handle_unknown="ignore"`, causing predictions for uncommon brands to revert toward general `year`/`odometer`/`fuel`/`transmission` averages rather than brand-specific pricing.

*(Add exact observed residual observations and MAE/RMSE percentage improvement over the median baseline here once `outputs/metrics/metrics.json` is generated.)*

### 7. Limitations and Ethical/Practical Caveats

1. **Asking Prices vs. Actual Transaction Prices:** Craigslist prices represent seller *asking* prices rather than confirmed final sale prices. Advertised prices include negotiation padding, dealer markups, occasional down-payment placeholders, and unsold over-priced listings, so app outputs must be presented only as illustrative listing estimates.
2. **Omitted Predictive Variables:** Important real-world valuation factors—specific model/trim, mechanical and cosmetic condition, accident/title history (clean vs. salvage), drivetrain (AWD/4WD/FWD), engine cylinders, and local market geography—are intentionally excluded from the 5-feature contract to keep the user interface simple, which bounds maximum achievable $R^2$.
3. **Dataset Biases and Temporal Drift:** Historical Craigslist snapshots reflect U.S. regional markets and past macroeconomic conditions (such as pre- or post-pandemic used-car supply shocks). Estimates can drift from current local market prices and may reflect geographic or demographic listing skews.
4. **Sampling Cap and Tree-Ensemble Extrapolation:** Capping training at `max_rows = 100000` trades a portion of available listings for fast local reproducibility. Additionally, Random Forests average training leaf values and therefore cannot extrapolate above the maximum or below the minimum training prices (e.g., brand-new future model years outside the historical training range).
5. **Artifact Trust Boundary:** Serialized `.joblib` files use Python pickle semantics under the hood; `src/modeling/inference.py` must only load locally generated, trusted repository artifacts.
