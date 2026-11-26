# Dataset Source and Description

## Project
- Course/project: ARTI303 – Craigslist Car Price Predictor
- Responsible member: M1 – Data Engineering
- Source file: `data/raw/vehicles.csv`
- Reported local file size: approximately 1.35 GB

## Dataset source and license
- Dataset page / download URL: **To be added after checking the original download page**
- Publisher/uploader: **To be confirmed**
- License / terms of use: **To be confirmed from the original dataset page**

The source URL and license were not present in the evidence available while preparing this document. Do not claim a license or unrestricted reuse until the original source page is checked. Add the exact URL, license name, and any attribution requirements before final submission.

## Dataset structure observed
The CSV inspection reported:
- **426,880 rows**
- **26 columns**

The observed columns were:
`id`, `url`, `region`, `region_url`, `price`, `year`, `manufacturer`, `model`, `condition`, `cylinders`, `fuel`, `odometer`, `title_status`, `transmission`, `VIN`, `drive`, `size`, `type`, `paint_color`, `image_url`, `description`, `county`, `state`, `lat`, `long`, `posting_date`.

## Fields used by the project
The team plan specifies the following six fields for the model handoff:

| Field | Description | Role / unit |
|---|---|---|
| `year` | Vehicle model year | Numeric feature |
| `odometer` | Vehicle mileage | Numeric feature; team plan specifies miles |
| `manufacturer` | Vehicle manufacturer | Categorical feature |
| `fuel` | Fuel type | Categorical feature |
| `transmission` | Transmission type | Categorical feature |
| `price` | Vehicle listing price | Target; team plan specifies USD |

The USD and miles units should be confirmed against the original dataset page or metadata before final submission.

## Initial missing-value inspection
The missing-value scan reported these counts for selected fields:

| Field | Missing values |
|---|---:|
| `year` | 1,205 |
| `manufacturer` | 17,646 |
| `model` | 5,277 |
| `condition` | 174,104 |
| `cylinders` | 177,678 |
| `fuel` | 3,013 |
| `odometer` | 4,400 |
| `title_status` | 8,242 |
| `transmission` | 2,556 |
| `VIN` | 161,042 |
| `drive` | 130,567 |
| `size` | 306,361 |
| `type` | 92,858 |
| `paint_color` | 130,203 |
| `county` | 426,880 |
| `lat` | 6,549 |
| `long` | 6,549 |

The scan also reported no missing values in `id`, `url`, `region`, `region_url`, `price`, and `state`.

## Raw-data handling
The raw dataset is kept locally at `data/raw/vehicles.csv`. It is not committed to GitHub because of its large size; team members download their own copy separately, as instructed by the team.

The raw source file should remain unchanged. Processed data is written to `data/processed/vehicles_clean.csv`.

## Verification items before final submission
1. Add the exact original dataset URL.
2. Confirm the publisher/uploader and license/usage terms.
3. Verify price and odometer units against the source metadata.
4. Confirm the six-field output schema and cleaning thresholds with M2/M3.
