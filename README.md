# Mtaa Property Intelligence

A Kenyan property-market demo with a FastAPI service, a responsive dashboard, a CSV-based data pipeline, and a scikit-learn price estimator. Property locations, features, and KES prices are simulated for product exploration and are not real listings or professional valuations. The bundled starter listings are ready to use; local refreshes can draw product IDs from DummyJSON.

## Run locally

1. Create and activate a virtual environment.
2. Install dependencies: `python -m pip install -r requirements.txt`
3. Start the app from the repository root: `python -m uvicorn api.main:app --reload`
4. Open <http://127.0.0.1:8000>.
5. Choose **Import demo listings** to fetch the sample product catalog, build the clean dataset, and train the model.

Importing refreshed listings needs an internet connection to `dummyjson.com`. Once imported, the dashboard reads the local CSV files and predictions use the saved model without requiring a database.

## Deploy to Vercel

The repository is configured for Vercel's FastAPI/Python runtime. The `pyproject.toml` selects `api.main:app` and declares the pinned runtime dependencies, matching `requirements.txt` for local installs. Vercel serves the dashboard's `/static` assets through its CDN, and `vercel.json` explicitly includes the demo CSVs and trained model in the function bundle. Python 3.12 and pinned dependencies keep the serialized estimator compatible at runtime.

1. Import this repository into Vercel, or install the Vercel CLI and run `vercel` from the repository root.
2. Keep the project root at the repository root. Leave the Build Command and Output Directory unset; Vercel detects the FastAPI app and installs the pinned dependencies declared in `pyproject.toml`.
3. Deploy with `vercel --prod` or enable Git deployments in the Vercel project.
4. Check `/api/health` on the deployed URL. The bundled demo data and trained model are available immediately.

Vercel function filesystems are temporary, so **Import demo listings** is intentionally read-only in production; its API returns a clear `503` instead of suggesting refreshed data was persisted. To publish updated listings, run the scraper, ETL, and trainer locally, commit `data/properties_raw.csv`, `data/properties_clean.csv`, and `models/property_price_model.joblib`, then redeploy. Configure a persistent database or object store before enabling in-production data refresh.

## Data and model commands

- `python scraper/property_api_scraper.py` downloads the simulated raw listings to `data/properties_raw.csv`.
- `python -m models.property_etl` validates and enriches them into `data/properties_clean.csv`.
- `python -m models.train_property_model` trains and saves `models/property_price_model.joblib`.
- `python -m models.time_series_forecast` writes a 30-day illustrative price forecast.
- `python -m models.location_clustering` groups listings by price and floor area.
- `python -m models.train_xgboost` optionally trains the alternate XGBoost estimator.

The equivalent end-to-end import and training operation is available in the dashboard locally. The API also exposes interactive documentation at `/docs` and health status at `/api/health`.
Install `xgboost` separately with `python -m pip install xgboost` to use the optional XGBoost trainer. Install `SQLAlchemy` with `python -m pip install SQLAlchemy` and configure `DATABASE_URL` to use the optional SQL database helper.

## Tests

Run the scraper, ETL, model, API-handler, forecast, and clustering checks with `python -m unittest discover -s tests -v`.

## Optional SQL storage

`database.db.DatabaseManager` is optional. Set `DATABASE_URL` to a SQLAlchemy connection URL before creating a database manager. The dashboard and training workflow do not require PostgreSQL.
