# Merge-safe file ownership and interface contracts

**Rule:** Every source file has one designated editing owner. Code review is cross-functional, but do not rewrite another owner's file in the same PR. All four members contribute their own branches and commits. No direct commits to `main`.

| Member | Only edits these folders | Depends on | Supplies |
|---|---|---|---|
| M1 | `src/data_workflow/`, `tests/data/`, personal sections | Kaggle CSV + config | cleaned CSV, EDA image, data-summary JSON |
| M2 | `src/modeling/`, `tests/modeling/`, personal sections | cleaned CSV/schema | joblib model, metrics JSON, evaluation narrative |
| M3 | `src/interface/`, `tests/interface/`, personal sections | agreed input fields, model & image | Streamlit app UI, screenshots, validation demonstration |
| M4 | `src/shared/`, `.github/`, root entry/config, `plan/`, `docs/`, final report/slides | all member outputs | quality checks, merged final deliverables, setup instructions |

`src/shared/config.py` and `config.yaml` are M4-controlled. Request any schema change as an issue (and obtain M1/M2/M3 agreement) before it is merged. `app.py` remains a 5-line stable entry point; M3 only edits `src/interface/ui.py`.

## Contracts (freeze by end of Week 1)

* Raw input: `data/raw/vehicles.csv`; at least `price,year,odometer,manufacturer,fuel,transmission` must exist.
* Processed output: `data/processed/vehicles_clean.csv`; columns `year,odometer,manufacturer,fuel,transmission,price`; numerics as numbers, categorical strings lowercased. Empty/invalid fields handled per implementation. `price` is USD asking price, `odometer` is in miles.
* Model: serialized sklearn `Pipeline`, accepts exactly `year,odometer,manufacturer,fuel,transmission`, returns one finite predicted price in USD; persisted `outputs/models/price_pipeline.joblib`.
* Metrics JSON: `rows,test_rows,mae,rmse,r2,baseline_mae,baseline_rmse`; no estimated figures in docs until trained.
* Chart: `outputs/figures/price_distribution.png`; local generated result.

## Handoff acceptance

M1 -> M2: CSV opens, required columns nonnull/valid, row counts in summary. M2 -> M3: model loads from trusted local artifact, sample frame accepted, metrics present. M3 -> M4: form predicts, rejects negative mileage, chart present or useful fallback. M4 -> group: all tests pass, commands rerun in clean environment, sources and limitations documented.
