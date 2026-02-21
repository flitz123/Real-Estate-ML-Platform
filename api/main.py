from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI(title="Property Price Prediction API")

model = joblib.load("models/xgboost_property_model.pkl")


@app.get("/")
def root():
    return {"message": "Property Price API Running"}


@app.post("/predict")
def predict(
    bedrooms: int,
    bathrooms: int,
    square_meters: float,
    rating: float,
    stock: int,
    price_per_sqm: float,
    bed_bath_ratio: float,
    location_Nairobi: int = 0,
    location_Mombasa: int = 0,
    location_Nakuru: int = 0,
    location_Kisumu: int = 0
):
    data = pd.DataFrame([{
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "square_meters": square_meters,
        "rating": rating,
        "stock": stock,
        "price_per_sqm": price_per_sqm,
        "bed_bath_ratio": bed_bath_ratio,
        "location_Nairobi": location_Nairobi,
        "location_Mombasa": location_Mombasa,
        "location_Nakuru": location_Nakuru,
        "location_Kisumu": location_Kisumu
    }])

    prediction = model.predict(data)
    return {"predicted_price": float(prediction[0])}
