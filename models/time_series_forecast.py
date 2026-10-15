from pathlib import Path

import pandas as pd
from sklearn.linear_model import LinearRegression

from models.property_etl import PROCESSED_DATA_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORECAST_PATH = PROJECT_ROOT / "data" / "price_forecast.csv"


def forecast_prices(data_path=PROCESSED_DATA_PATH, output_path=FORECAST_PATH, days=30):
    data_path = Path(data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Clean property data not found at {data_path}.")
    data = pd.read_csv(data_path, parse_dates=["listing_date"])
    data = data.dropna(subset=["listing_date", "price"]).sort_values("listing_date")
    if data["listing_date"].nunique() < 2:
        raise ValueError("At least two distinct listing dates are needed for a forecast.")
    if days < 1:
        raise ValueError("Forecast days must be greater than zero.")

    first_day = data["listing_date"].min().normalize()
    x = (data["listing_date"] - first_day).dt.days.to_frame(name="days")
    model = LinearRegression().fit(x, data["price"])
    future_dates = pd.date_range(data["listing_date"].max().normalize() + pd.Timedelta(days=1), periods=days)
    future_x = (future_dates - first_day).days.to_frame(index=False, name="days")
    forecast = pd.DataFrame(
        {"date": future_dates, "predicted_price": model.predict(future_x)}
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    forecast.to_csv(output_path, index=False)
    return forecast


if __name__ == "__main__":
    predictions = forecast_prices()
    print(f"Created {len(predictions)} daily forecasts at {FORECAST_PATH}")
