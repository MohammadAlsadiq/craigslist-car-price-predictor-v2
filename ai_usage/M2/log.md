# M2 AI Usage Evidence

*Student: Ibrahim Alhajjaj (M2 — Machine Learning Engineering)*

## Prompt & Assistance Log

| Date | Tool | Prompt | Suggested content | Independent checks/corrections | Implemented by student |
|---|---|---|---|---|---|
| 2026-11-24 | Antigravity AI coding assistant | Complete M2 deliverables end-to-end across `src/modeling/`, `tests/modeling/`, `docs/member_notes/M2/`, `reports/sections/M2/`, `slides/sections/M2/`, and `ai_usage/M2/`, strictly respecting M2 file ownership, leakage prevention, M3 inference contract, and no fabricated metrics. | Refactored `src/modeling/train.py` into modular functions (`build_pipeline`, `load_cleaned_data`, `train_and_evaluate`, `save_artifacts`, `run_training`) with post-split `ColumnTransformer`, `baseline_r2` + split row counts in `metrics.json`, and corrected CLI hint (`python -m src.data_workflow.pipeline`). Hardened `src/modeling/inference.py` (`validate_car`, `load_model`, `predict_price`) with finite/bool guards, unseen-category safety, and corrected missing-model error hint (`python -m src.modeling.train`). Expanded `tests/modeling/test_training.py` and `tests/modeling/test_inference.py` with leakage, identical-test-row baseline comparison, unseen-manufacturer, artifact reload, and missing-file tests. Drafted M2 notes, report section, and slide bullets with `[PENDING]` placeholders because `data/processed/vehicles_clean.csv` is not yet present. | Verified all edits stayed strictly within M2-owned paths via `git status`; verified `src/interface/ui.py` and `tests/interface/test_contract.py` compatibility without modifying non-M2 files; executed full `pytest` suite to confirm all M1/M2/M3 tests pass; verified zero fabricated metrics exist in docs/report/slides. | Ibrahim Alhajjaj to review all generated code, tests, and documentation; run `python -m src.modeling.train` locally once M1 hands off `data/processed/vehicles_clean.csv`; verify real `outputs/metrics/metrics.json` values and replace `[PENDING]` placeholders in report and slides before final submission. |

## Human Verification Checklist (Before Final Submission)

- [ ] Review `src/modeling/train.py` and `src/modeling/inference.py` line-by-line and be prepared to explain every design decision (post-split `ColumnTransformer`, `OneHotEncoder(handle_unknown="ignore")`, `DummyRegressor` vs. `RandomForestRegressor`, `max_rows` cap).
- [ ] Confirm `data/processed/vehicles_clean.csv` schema handoff with M1 and run `python -m src.modeling.train` to generate real local artifacts (`outputs/models/price_pipeline.joblib` and `outputs/metrics/metrics.json`).
- [ ] Inspect real MAE, RMSE, $R^2$, and row counts in `outputs/metrics/metrics.json` and replace every `[PENDING]` placeholder in `reports/sections/M2/section.md` and `slides/sections/M2/slides.md`.
- [ ] Confirm with M3 that `streamlit run app.py` loads the trained `outputs/models/price_pipeline.joblib` and handles unseen manufacturers and missing-artifact warnings cleanly.
- [ ] Confirm with M4 that the `m2-complete-modeling` branch merges cleanly with zero edits outside M2-owned folders.
