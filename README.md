# Craigslist Used Car Price Predictor | ARTI303

Four-person university project. **The instructor's guidelines and rubric** are in `docs/reference/`. Submission requires GitHub + project report PDF + AI Usage Report (PDF or Markdown) + slides (PPTX or PDF) + working Streamlit app. Topic must be approved/registered; do not assume approval from the starter repository.

## Run locally

1. Download Craigslist Used Car Prices from Kaggle. Place `vehicles.csv` in `data/raw/` (not committed; confirm license). The dataset has not been bundled.
2. Create a virtual environment, then run:

```bash
python -m pip install -r requirements.txt
python -m src.data_workflow.pipeline
python -m src.modeling.train
python -m pytest -q
streamlit run app.py
```

The first command after installing performs Load -> Validate -> Clean -> Explore -> Visualize -> Transform -> Export -> Summary. Scikit-learn categorical one-hot encoding is **fitted only on the training subset** within the saved model pipeline. The training command compares a median baseline with a Random Forest, saves MAE/RMSE/R² and `joblib` model. The app accepts year/odometer/manufacturer/fuel/transmission, guards invalid inputs, and displays a saved EDA chart.

## Who edits what?

- **M1 Data Engineer:** `src/data_workflow/`, `tests/data/`, `reports/sections/M1/`, `slides/sections/M1/`, `ai_usage/M1/`.
- **M2 ML Engineer:** `src/modeling/`, `tests/modeling/`, `reports/sections/M2/`, `slides/sections/M2/`, `ai_usage/M2/`.
- **M3 Streamlit Engineer:** `src/interface/`, `tests/interface/`, `reports/sections/M3/`, `slides/sections/M3/`, `ai_usage/M3/`.
- **M4 QA/Documentation/Integration:** `src/shared/`, `.github/`, `docs/`, `plan/`, `reports/final/`, `slides/final/`, `ai_usage/M4/` and final integration of `README.md`, `config.yaml`, `app.py`.

**Don't edit another member's owned source file**; propose a change in their PR or open an issue. M4 merges completed outputs only after review; all four members push commits. Use separate branches `feature/m1-data`, `feature/m2-model`, `feature/m3-app`, `feature/m4-qa-docs`.

## Handoff contracts

1. M1 produces `data/processed/vehicles_clean.csv`, `outputs/figures/price_distribution.png`, `outputs/reports/data_summary.json` after reading `data/raw/vehicles.csv`.
2. M2 consumes cleaned CSV and agreed feature schema; produces `outputs/models/price_pipeline.joblib` + `outputs/metrics/metrics.json`.
3. M3 consumes saved model and visualization; runs inference via `src.modeling.inference` and validates inputs before showing prediction.
4. M4 verifies full reproducible run and compiles final report and presentation from member-owned sections. **Generated outputs are local by default and ignored by Git.** Distribute screenshots/aggregated summaries or permitted small examples; do not commit the Kaggle dataset or big models without approval.

See `plan/ARTI303_Weekly_Team_Plan.pdf` for editable member names, per-week checklists, required resources and exact handoffs. See `docs/OWNERSHIP_AND_CONTRACTS.md` and `docs/RUBRIC_MATRIX.md`.

## Limitations

Listings are advertised prices rather than confirmed sales. Missing values, duplicate postings, historic data and regional biases affect estimates. `max_rows` in `config.yaml` caps training for affordable local compute; disclose any sampling. Never treat a fabricated sample result as a real evaluation. Only load trusted joblib artifacts. Any use of generative AI must be logged and verified individually.
