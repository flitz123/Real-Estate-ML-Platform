import joblib
import pandas as pd

model = joblib.load("models/xgboost_property_model.pkl")

sample_property = pd.DataFrame([{
    "bedrooms": 3,
    "bathrooms": 2,
    "square_meters": 150,
    "rating": 4.5,
    "stock": 20,
    "price_per_sqm": 100000,
    "bed_bath_ratio": 1.5,
    "location_Mombasa": 0,
    "location_Nairobi": 1,
    "location_Nakuru": 0,
    "location_Kisumu": 0
}])

prediction = model.predict(sample_property)
print("Predicted Price:", prediction[0])
