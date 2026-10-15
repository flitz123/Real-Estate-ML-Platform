# Mtaa Property Intelligence

A local Kenyan property-market demo with a FastAPI service, a responsive dashboard, a CSV-based data pipeline, and a scikit-learn price estimator. The listing source is DummyJSON; property locations, features, and KES prices are simulated for product exploration and are not real listings or professional valuations.

## Run locally

1. Create and activate a virtual environment.
2. Install dependencies: `python -m pip install -r requirements.txt`
3. Start the app from the repository root: `uvicorn api.main:app --reload`
4. Open <http://127.0.0.1:8000>.
5. Choose **Import demo listings** to fetch the sample product catalog, build the clean dataset, and train the model.

The first import needs an internet connection to `dummyjson.com`. Once imported, the dashboard reads the local CSV files and predictions use the saved model without requiring a database.

## Data and model commands

- `python scraper/property_api_scraper.py` downloads the simulated raw listings to `data/properties_raw.csv`.
- `python -m models.property_etl` validates and enriches them into `data/properties_clean.csv`.
- `python -m models.train_property_model` trains and saves `models/property_price_model.joblib`.
- `python -m models.time_series_forecast` writes a 30-day illustrative price forecast.
- `python -m models.location_clustering` groups listings by price and floor area.
- `python -m models.train_xgboost` optionally trains the alternate XGBoost estimator.

The equivalent end-to-end import and training operation is available in the dashboard. The API also exposes interactive documentation at `/docs` and health status at `/api/health`.

## Tests

Run the scraper, ETL, model, API-handler, forecast, and clustering checks with `python -m unittest discover -s tests -v`.

## Optional SQL storage

`database.db.DatabaseManager` is optional. Set `DATABASE_URL` to a SQLAlchemy connection URL before creating a database manager. The dashboard and training workflow do not require PostgreSQL.
