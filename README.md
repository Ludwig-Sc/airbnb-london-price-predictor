# Airbnb London — Price Prediction (Final Project)

Final version of the *Data Science for Business* (KIT, 1st semester) case study:
predicting nightly Airbnb prices in London from listing characteristics, and
serving the result through an interactive Streamlit app.

This repository is a **cleaned-up, restructured copy** of the final project
folder, which mixed 9 differently-named script files, three near-duplicate
model-run folders, and several Git LFS pointer stubs that had never been
pulled. Nothing about the analysis logic was changed — scripts were renamed
by pipeline stage, given a single consistent "run from the repo root"
convention, and the two file paths the Streamlit app needs were updated to
match the new layout. Everything else (what each script actually computes) is
untouched. See [Provenance & known issues](#provenance--known-issues) for the
full list of what was touched and what real problems were found underneath.

## Project history

This is the second, expanded phase of a project that started earlier:

1. **Solo exploration** — first look at the raw Inside Airbnb files (`.info()`/`.describe()`/null counts).
2. **Phase 1 — cleaning & baseline pipeline**: data cleaning, category-split CSVs, a bedroom/price deep-dive, and a first Linear/Ridge/Lasso model. Reconstructed separately at [`airbnb-london-analysis`](https://github.com/Ludwig-Sc/airbnb-london-analysis).
3. **Phase 2 — this repo**: adds amenity feature engineering, geospatial analysis, an ID-joinable predictor table, a 6-model × 2-feature-set comparison, and the Streamlit price predictor. Distilled from the original working repo, [`DataScienceforBusinessnew`](https://github.com/Ludwig-Sc/DataScienceforBusinessnew).

Scripts `01`–`03` and `05` below are carried over from phase 1 essentially
unchanged (only comments were translated/reformatted between the two
projects) — see the phase 1 repo for their original context.

## Data

Source: the [Inside Airbnb](http://insideairbnb.com/) London snapshot
(`listings.csv`, `calendar.csv`, `reviews.csv`). Raw files are **not**
included (the listings file alone is ~195 MB) — see `data/raw/README.md`.

Processed data actually checked into `data/processed/` (real data, not
placeholders — see known issues for why that distinction matters here):

| File | Rows | Purpose |
|---|---|---|
| `category_csvs/` (11 files) | ~62k listings each | Cleaned listings split by topic (price, listing basics, location, host, booking, reviews, amenities, etc.) |
| `predictor_dataset.csv` | 61,950 | Earliest bedroom-focused predictor table (no `id` column) |
| `predictor_dataset_with_id.csv` | 61,351 | Same idea, `id`-joinable, used to merge with amenities/category data |
| `amenities_reduced_grouped.csv` | 61,350 | 23 grouped amenity flags (`has_wifi`, `has_kitchen`, …) per listing |
| `model_dataset_used_run2.csv` | 60,739 | Final, fully merged & cleaned table (67 columns) the winning model was trained on |

`category_csvs/*.csv`, `model_dataset_used_run2.csv`, and `models/*.joblib`
are tracked via **Git LFS** (`.gitattributes`) because they're large —
run `git lfs install` once, then clone/pull normally.

## Pipeline

Run everything from the repository root, e.g. `python scripts/01_clean_data.py`.

| Step | Script | What it does |
|---|---|---|
| 1 | `01_clean_data.py` | Cleans raw `listings.csv`: booleans → 1/0, strips `%`/`$` formatting, normalizes dates, drops empty columns. |
| 2 | `02_create_categories.py` | Splits cleaned listings into 9 topical CSVs + a column-coverage overview. |
| 3 | `03_aggregate_reviews_calendar.py` | Builds per-listing review and calendar/occupancy summaries. *(Not wired into later steps — see known issues.)* |
| 4 | `04_build_predictor_dataset.py` | `PredictorWithIDCreator`: merges price + listing basics + location on `id`, filters price outliers (top 1%), engineers `is_studio`/`bedrooms_log`. Outputs `predictor_dataset_with_id.csv`. |
| 5 | `05_category_eda_and_baseline_model.py` | Full EDA (price distribution, neighbourhood/room-type medians, correlation heatmap, superhost premium) + a Linear/Ridge/Lasso baseline on `log(price)`. |
| 6 | `06_geo_analysis.py` | Builds a price heatmap and a borough choropleth (Folium), computes distance-to-center (Haversine) and 8 K-Means geo-clusters, exports `london_geo_features.csv`. |
| 7 | `07_amenities_price_analysis.py` | `AmenitiesPriceAnalyzer`: parses the raw `amenities` list per listing into a full binary matrix and ranks amenities by median price impact. *(This is not the script that produced the grouped `amenities_reduced_grouped.csv` — see known issues.)* |
| 8 | `08_compare_prediction_models.py` | `AirbnbPredictionModelRunner`: trains Dummy/Ridge/ElasticNet/Random Forest/Extra Trees/HistGradientBoosting on two feature sets (Run 1: predictor+amenities; Run 2: +category features), with hyperparameter search, CV, feature importance, and saved models per run. |
| — | `app/price_predictor_app.py` | Streamlit app serving the final Run 2 model interactively. |

## Model evolution

Three model-comparison iterations exist in the project history, each on a
different feature set (best model shown; £ = test-set price):

| Iteration | Features | Best model | Test MAE | Test RMSE | Test R² |
|---|---|---|---|---|---|
| Early single run *(script lost)* | predictor table only, exact features unknown | Random Forest | £45.31 | £88.41 | 0.641 |
| Run 1 | predictor + amenities (31 features) | Random Forest | £49.14 | £80.28 | 0.616 |
| **Run 2 (final)** | predictor + amenities + category features (65 features) | **HistGradientBoosting** | **£39.37** | **£65.28** | **0.746** |

All numbers are the project's own recorded results (`overall_best_model_summary.json`,
`early_model_outputs/tables/best_model_summary.json` — real files, not
regenerated in this session; this machine can't currently run the pipeline,
see known issues). Run 1 scoring slightly worse than the earlier, simpler run
is odd but unverifiable — the earlier run's script no longer exists, so its
exact feature engineering and price-outlier handling are unknown.

## The app

`app/price_predictor_app.py` loads the Run 2 `HistGradientBoostingRegressor`
and lets a user configure a listing (bedrooms, room/property type,
neighbourhood, reviews, host features, grouped amenities). Selecting a
neighbourhood auto-fills its median latitude/longitude (with an optional
manual override) so location inputs stay internally consistent. The
predicted price comes with a 90% local prediction interval built from
residuals of the nearest 1,000 test-set predictions by predicted price.

```bash
pip install -r requirements.txt
python -m streamlit run app/price_predictor_app.py
```

## Provenance & known issues

Found while reconstructing this repo — worth knowing before extending it:

- **Git LFS pointers were unhydrated in the working folder this was built
  from.** `category_csvs/`, `predictor_dataset_with_id.csv`, the trained
  `.joblib` models, and the model's dataset/predictions tables were all just
  ~130-byte LFS pointer stubs locally, not real data. The real files were
  pulled by cloning [`Ludwig-Sc/DataScienceforBusinessnew`](https://github.com/Ludwig-Sc/DataScienceforBusinessnew)
  with `git-lfs` installed. This repo keeps tracking the same large files via
  LFS (see `.gitattributes`) so a normal clone here actually works.
- **Three scripts are missing; only their outputs survive**, all showing the
  same pattern of lost/overwritten source code:
  - `amenities_reduced_grouped.csv` (23 grouped amenity flags, used by the
    final model) — no script producing it exists anywhere in the project.
    The one amenities script that does survive, `07_amenities_price_analysis.py`,
    does something different (a full per-amenity binary matrix + a price-impact
    ranking CSV, no grouping, no charts).
  - `outputs/amenities_figures/*.png` (7 charts on amenity price impact) —
    same story; `07_amenities_price_analysis.py` doesn't plot anything.
  - `outputs/model_results/early_single_run/` (the first, simplest model
    comparison: Dummy/Ridge/Random Forest/Extra Trees on the predictor table
    alone) — superseded by `08_compare_prediction_models.py`, a clear rewrite
    of the same idea as a proper class with two feature-set runs and six
    models, but the original script is gone.
- **`04_build_predictor_dataset.py` has a real bug**: it defines a
  `needed_reviews` column list (`review_scores_rating`, `number_of_reviews`)
  but never merges the reviews category table into the dataframe, so those
  two columns silently never make it into `predictor_dataset_with_id.csv`
  despite the code appearing to request them.
- **Only the final model-comparison run is kept.** The comparison script was
  actually run three times (`20260715_142444`, `20260715_144328`,
  `20260720_233836`) with near-identical output; only the last, most complete
  run (the only one with per-model prediction CSVs and permutation
  importance) is kept here to avoid duplicate multi-MB output. The earlier
  two remain in the original repo if needed.
- **Inconsistent working-directory assumptions, now standardized.** Originally
  every script assumed a different current directory — some used `../category_csvs`,
  some `category_csvs`, `06_geo_analysis.py` used bare filenames (meaning
  "run me from inside the category_csvs folder"), and `08`'s class defaults
  pointed at a hardcoded `/DataScienceforBusiness-1/...` path that was only
  ever overridden, never actually used. This repo standardizes all scripts to
  run from the repository root — the only behavioral (path-only) change made
  beyond file moves and renames.
- **`06_geo_analysis.py` depends on a live network call** to a third-party
  GitHub-hosted GeoJSON file for London borough boundaries at runtime.
- **Mixed decimal/locale conventions** persist across scripts (`;`-separated
  European-decimal vs. plain comma-separated US-decimal CSVs), same as in
  phase 1 — each script re-detects and re-normalizes this itself.
- **This environment's Python install couldn't run the pipeline** (`numpy`
  2.x / `pyarrow` ABI mismatch breaks `pandas`/`scikit-learn` imports), so
  nothing here was re-executed to regenerate numbers or plots — all reported
  metrics and figures are the project's own recorded output.

## Limitations

- Very expensive luxury listings are harder to predict accurately.
- Availability features are proxies, not exact booking/occupancy rates.
- External factors (events, seasonality, tourism demand, competitor pricing)
  aren't modeled.
- The model reflects historical data and won't automatically track market
  changes.

## Repository layout

```
airbnb-london-price-predictor/
├── data/
│   ├── raw/                  # put listings.csv / reviews.csv / calendar.csv here (not tracked)
│   └── processed/
│       ├── category_csvs/            # LFS — 9 topical CSVs + overview
│       ├── predictor_dataset.csv
│       ├── predictor_dataset_with_id.csv
│       ├── amenities_reduced_grouped.csv
│       └── model_dataset_used_run2.csv   # LFS
├── scripts/
│   ├── 01_clean_data.py
│   ├── 02_create_categories.py
│   ├── 03_aggregate_reviews_calendar.py
│   ├── 04_build_predictor_dataset.py
│   ├── 05_category_eda_and_baseline_model.py
│   ├── 06_geo_analysis.py
│   ├── 07_amenities_price_analysis.py
│   └── 08_compare_prediction_models.py
├── app/
│   └── price_predictor_app.py
├── models/
│   └── best_overall_estimator.joblib      # LFS — final HistGradientBoosting model
├── outputs/
│   ├── eda_figures/           # script 05 + 06 charts
│   ├── amenities_figures/     # 7 amenity price-impact charts (script lost, see known issues)
│   ├── geo_maps/              # Folium heatmap + choropleth HTML
│   └── model_results/
│       ├── early_single_run/  # first, simplest model comparison (script lost)
│       └── final_run/         # the kept 20260720_233836 comparison run
│           └── plots/run1_predictor_plus_amenities/, run2_predictor_plus_amenities_plus_category/
├── requirements.txt
├── .gitattributes             # Git LFS tracking for large files
└── README.md
```

## Project Context

This project was created for the course **Data Science for Business** (KIT,
1st semester), focused on building a business-oriented price prediction
workflow for Airbnb listings in London.
