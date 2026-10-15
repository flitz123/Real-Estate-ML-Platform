from pathlib import Path
from typing import Literal

import joblib
import os
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
import requests

from models.model_pipeline import FEATURES, MODEL_PATH, train_property_model
from models.property_etl import PROCESSED_DATA_PATH, RAW_DATA_PATH, transform_property_data
from scraper.property_api_scraper import PropertyAPIScraper


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WEB_DIR = PROJECT_ROOT / "frontend"
SYNC_ENABLED = os.getenv("VERCEL") != "1"
app = FastAPI(
    title="Mtaa — Property Intelligence",
    description="Kenyan property market insights and price estimates.",
    version="1.0.0",
)
app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


class PredictionRequest(BaseModel):
    bedrooms: int = Field(ge=1, le=20)
    bathrooms: int = Field(ge=1, le=20)
    square_meters: float = Field(gt=0, le=100_000)
    rating: float = Field(ge=0, le=5)
    stock: int = Field(ge=0)
    location: Literal["Nairobi", "Eldoret", "Mombasa", "Kisumu", "Nakuru"]


def _load_market_data():
    if not RAW_DATA_PATH.is_file():
        return pd.DataFrame()
    data = pd.read_csv(RAW_DATA_PATH)
    if data.empty:
        return data
    if "square_meters" not in data and "square_metres" in data:
        data = data.rename(columns={"square_metres": "square_meters"})
    if "location" not in data and "locations" in data:
        data = data.rename(columns={"locations": "location"})
    return data


def _load_model_bundle():
    if not MODEL_PATH.is_file():
        raise HTTPException(
            status_code=503,
            detail="No trained model is available. Import demo listings to train one.",
        )
    return joblib.load(MODEL_PATH)


@app.get("/", include_in_schema=False)
def home():
    return FileResponse(WEB_DIR / "index.html")


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "model_ready": MODEL_PATH.is_file(),
        "market_data_ready": RAW_DATA_PATH.is_file(),
        "sync_enabled": SYNC_ENABLED,
    }


@app.get("/api/market")
def market():
    data = _load_market_data()
    if data.empty:
        return {
            "total_listings": 0,
            "average_price": None,
            "median_price": None,
            "average_size": None,
            "locations": 0,
            "price_by_location": [],
            "listings": [],
            "model_ready": MODEL_PATH.is_file(),
            "mae": None,
            "sync_enabled": SYNC_ENABLED,
        }

    for column in ("price", "square_meters"):
        if column in data:
            data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(subset=["price", "location"])
    prices = data["price"]
    grouped = (
        data.groupby("location", as_index=False)
        .agg(average_price=("price", "mean"), listings=("price", "size"))
        .sort_values("average_price", ascending=False)
    )
    listing_columns = [
        column
        for column in (
            "property_id",
            "title",
            "price",
            "bedrooms",
            "bathrooms",
            "square_meters",
            "location",
            "rating",
        )
        if column in data
    ]
    listings = data[listing_columns].iloc[::-1]
    listings = listings.astype(object).where(pd.notna(listings), None)
    mae = None
    if MODEL_PATH.is_file():
        mae = _load_model_bundle().get("mae")

    return {
        "total_listings": int(len(data)),
        "average_price": float(prices.mean()),
        "median_price": float(prices.median()),
        "average_size": (
            float(data["square_meters"].mean())
            if "square_meters" in data and data["square_meters"].notna().any()
            else None
        ),
        "locations": int(data["location"].nunique()),
        "price_by_location": [
            {
                "location": row.location,
                "average_price": float(row.average_price),
                "listings": int(row.listings),
            }
            for row in grouped.itertuples(index=False)
        ],
        "listings": listings.to_dict(orient="records"),
        "model_ready": MODEL_PATH.is_file(),
        "mae": float(mae) if mae is not None else None,
        "sync_enabled": SYNC_ENABLED,
    }


@app.post("/api/market/sync")
def sync_market():
    if not SYNC_ENABLED:
        raise HTTPException(
            status_code=503,
            detail=(
                "Market refresh is disabled on Vercel because serverless filesystems "
                "are temporary. Update the demo dataset and model, then redeploy."
            ),
        )
    try:
        scraped = PropertyAPIScraper().scrape()
        transform_property_data()
        training = train_property_model()
    except requests.RequestException as error:
        raise HTTPException(
            status_code=502, detail=f"Could not reach the demo listing source: {error}"
        ) from error
    except (FileNotFoundError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    return {
        "message": "Demo listings imported and the price model was trained.",
        "imported_listings": len(scraped),
        "training_rows": training["training_rows"],
        "mae": training["mae"],
    }


@app.post("/api/predict")
def predict(request: PredictionRequest):
    bundle = _load_model_bundle()
    features = pd.DataFrame(
        [
            {
                "bedrooms": request.bedrooms,
                "bathrooms": request.bathrooms,
                "square_meters": request.square_meters,
                "rating": request.rating,
                "stock": request.stock,
                "location": request.location,
                "bed_bath_ratio": request.bedrooms / request.bathrooms,
            }
        ]
    )
    prediction = float(bundle["model"].predict(features[FEATURES])[0])
    return {
        "predicted_price": max(0, prediction),
        "currency": "KES",
        "model_mae": bundle.get("mae"),
    }
