# M1 – Data Engineering Report

## 1. Responsibility
M1 is responsible for the data engineering workflow: source/data verification, validation, cleaning, exploratory analysis, visualization, transformation, export, and summary generation.

M1-owned project paths:
- `src/data_workflow/`
- `tests/data/`
- `reports/sections/M1/`
- `slides/sections/M1/`
- `ai_usage/M1/`

## 2. Dataset overview
The source file is `data/raw/vehicles.csv`, reported to be approximately 1.35 GB. Initial inspection of the actual CSV reported **426,880 records** and **26 columns**.

The six fields specified by the team plan are:
`year`, `odometer`, `manufacturer`, `fuel`, `transmission`, and target `price`.

The team plan specifies price in USD and odometer in miles. These units still need confirmation against the original dataset metadata.

## 3. Data quality inspection
A missing-value scan was run over the source data. Examples of missing counts:
- `county`: 426,880
- `size`: 306,361
- `cylinders`: 177,678
- `condition`: 174,104
- `VIN`: 161,042
- `drive`: 130,567
- `paint_color`: 130,203
- `type`: 92,858
- `manufacturer`: 17,646
- `year`: 1,205
- `odometer`: 4,400

A separate validity check reported:
- Invalid prices (missing, zero, or negative): 32,895
- Invalid years outside the tested range: 0
- Negative odometer values: 0
- Duplicate IDs: 0

## 4. Pipeline implementation
The existing `src/data_workflow/pipeline.py` contains the eight stages required by the project plan:

1. Load
2. Validate
3. Clean
4. Explore
5. Visualize
6. Transform
7. Export
8. Summary

The pipeline uses the project configuration through `src.shared.config` and writes the cleaned dataset, price distribution figure, and JSON summary to the configured output paths.

## 5. Cleaning and execution results
The pipeline was successfully executed on the source CSV. The terminal log reported:

- Loaded rows: **426,880**
- Retained cleaned rows: **209,486**
- Median price: **USD 11,499**
- Mean price: **USD 15,509.89**
- Distinct manufacturers: **43**

The pipeline therefore removed **217,394** rows overall. This total must not be attributed only to invalid prices: the current `clean()` function also drops rows missing numeric fields and applies price, year, and odometer thresholds from `config.yaml`.

The current code fills missing categorical feature values with `unknown`, drops rows missing numeric features or target, removes duplicate rows, and applies configured range filters. The exact configured thresholds should be recorded from `config.yaml` before final submission.

## 6. Outputs generated
The successful run generated the following files, which were visible in the project folders:

- `data/processed/vehicles_clean.csv`
- `outputs/figures/price_distribution.png`
- `outputs/reports/data_summary.json`

These are the M1 data handoff artifacts for M2 and the EDA artifact for the project.

## 7. Limitations and items to confirm
- Verify the dataset source URL and license; these were not established in the available evidence.
- Verify USD and miles units against the original dataset metadata.
- Confirm the configured cleaning thresholds with M2/M3.
- Review whether dropping rows with missing `year` or `odometer` is the agreed strategy, rather than imputing them downstream.
- Confirm that the output CSV contains exactly the six agreed fields and that M2 can load it.

## 8. Conclusion
The existing M1 pipeline ran successfully and generated the three expected output artifacts. The observed cleaned dataset contains 209,486 rows from 426,880 source rows. Final sign-off depends on confirming the dataset citation/license, units, configuration thresholds, and the agreed data contract with M2/M3.
