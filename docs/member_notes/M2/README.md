# M2 Engineering Notes — Machine Learning Modeling & Inference

**Owner:** Ibrahim Alhajjaj (M2 — Machine Learning Engineering)  
**Owned Paths:** `src/modeling/`, `tests/modeling/`, `docs/member_notes/M2/`, `reports/sections/M2/`, `slides/sections/M2/`, `ai_usage/M2/`

---

## 1. Agreed Feature Schema & Target

All settings are loaded from `config.yaml` via `src.shared.config.load_config`. Training consumes **only** M1's cleaned output (`data/processed/vehicles_clean.csv`) and never reads `data/raw/vehicles.csv`.

| Column | Role | Type & Units | Cleaning / Normalization Expectation |
|---|---|---|---|
| `year` | Numeric input | Integer model year | Within `[1980, 2027]` (`config.yaml`) |
| `odometer` | Numeric input | Float mileage in **miles** | Within `[0, 500000]` miles (`config.yaml`) |
| `manufacturer` | Categorical input | Lowercase trimmed string | Unknown/unseen values handled safely at inference |
| `fuel` | Categorical input | Lowercase trimmed string | e.g., `gas`, `diesel`, `hybrid`, `electric`, `other` |
| `transmission` | Categorical input | Lowercase trimmed string | e.g., `automatic`, `manual`, `other` |
| `price` | Target | Float asking price in **USD** | Within `[500, 150000]` USD (`config.yaml`); listing price, not confirmed sale price |

---

## 2. Model & Inference Contract for M3 (`src/interface/ui.py`)

`src/modeling/inference.py` exposes the stable interface consumed by M3's Streamlit UI:

1. `validate_car(year, odometer, manufacturer, fuel, transmission, cfg) -> pd.DataFrame`
   - Validates finite numeric `year` (whole number in `[min_year, max_year]`) and `odometer` (`0 <= odometer <= max_odometer`), rejecting booleans, NaN/Inf, out-of-range values, and blank strings with `ValueError`.
   - Normalizes `manufacturer`, `fuel`, and `transmission` via `.strip().lower()`.
   - Returns a 1-row `pd.DataFrame` with exact column order `["year", "odometer", "manufacturer", "fuel", "transmission"]`.
2. `load_model(model_path) -> Pipeline`
   - Loads the local trusted artifact (`outputs/models/price_pipeline.joblib`).
   - Raises `FileNotFoundError` with an actionable message (`python -m src.modeling.train`) if the artifact has not been generated yet.
3. `predict_price(model_path, frame) -> float`
   - Validates that `frame` is a non-empty `pd.DataFrame` containing all 5 required columns, defensively normalizes categorical strings, loads the saved pipeline, and returns a finite non-negative float prediction in **USD**.
   - **Unseen categories:** Because `OneHotEncoder(handle_unknown="ignore")` is fitted inside the saved `Pipeline`, passing an unseen manufacturer (e.g., `"rivian"` or `"unknown_brand_xyz"`) encodes to all zeros for that feature and never crashes inference.
   - **Error handling alignment:** Raises `ValueError` for invalid input frames and `FileNotFoundError` for missing artifacts — both of which `src/interface/ui.py` explicitly catches and surfaces via `st.warning(...)`.

---

## 3. Split Configuration, Sample Cap & Leakage Prevention

Configuration values read from `config.yaml`:

- **Train/test split:** Single hold-out split with `test_size = 0.2` (80% train / 20% test) and `random_state = 42`.
- **Sample cap:** `max_rows = 100000`. If M1's cleaned CSV has more than `100,000` rows, `train_and_evaluate()` deterministically samples `100,000` rows with `random_state = 42` before splitting and records both `source_cleaned_rows` and `rows` (plus `sampled: true/false`) in `outputs/metrics/metrics.json`.
- **Post-split preprocessing rule (no data leakage):**
  - Numeric branch: `SimpleImputer(strategy="median")`
  - Categorical branch: `SimpleImputer(strategy="most_frequent")` -> `OneHotEncoder(handle_unknown="ignore")`
  - Both branches live inside a `ColumnTransformer` wrapped inside the scikit-learn `Pipeline` (`[("preprocess", preprocess), ("model", RandomForestRegressor(...))]`).
  - `.fit(X_train, y_train)` is called **only** after `train_test_split`, ensuring medians, modes, and one-hot vocabulary are learned exclusively from `X_train` and never see `X_test`.

---

## 4. Experiment Design: Median Baseline vs. Random Forest

Both models are fitted on the **exact same** `(X_train, y_train)` split and evaluated on the **exact same** `(X_test, y_test)` rows:

1. **Baseline:** `DummyRegressor(strategy="median")` — predicts the training-set median listing price for every test car. Serves as a sanity floor to verify the ML model adds real predictive value beyond guessing the typical price.
2. **Model:** `RandomForestRegressor(n_estimators=120, random_state=42, n_jobs=-1)` — captures non-linear depreciation curves over `year` and `odometer` as well as interactions with `manufacturer`, `fuel`, and `transmission`.
3. **Evaluation Metrics (saved to `outputs/metrics/metrics.json`):**
   - `mae`, `rmse`, `r2` (Random Forest on held-out test split)
   - `baseline_mae`, `baseline_rmse`, `baseline_r2` (Median DummyRegressor on identical held-out test split)
   - `rows`, `source_cleaned_rows`, `sampled`, `train_rows`, `test_rows`, `test_size`, `random_state`, `n_estimators`, `max_rows`

---

## 5. Run & Handoff Status

- **Current data status:** `data/processed/vehicles_clean.csv` is not yet present in this workspace (waiting on M1's cleaned dataset handoff).
- **Once M1 generates `data/processed/vehicles_clean.csv`, run:**
  ```bash
  python -m src.modeling.train
  python -m pytest -q
  ```
- This will create local artifacts `outputs/models/price_pipeline.joblib` and `outputs/metrics/metrics.json` (git-ignored per `.gitignore`), after which the `[PENDING]` tables in `reports/sections/M2/section.md` and `slides/sections/M2/slides.md` should be populated with the exact recorded test values.
