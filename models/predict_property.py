import joblib
import pandas as pd

from models.model_pipeline import FEATURES, MODEL_PATH


def predict_sample_property(
    bedrooms=3,
    bathrooms=2,
    square_meters=150,
    rating=4.5,
    stock=20,
    location="Nairobi",
):
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Train a model first."
        )

    bundle = joblib.load(MODEL_PATH)
    sample = pd.DataFrame(
        [
            {
                "bedrooms": bedrooms,
                "bathrooms": bathrooms,
                "square_meters": square_meters,
                "rating": rating,
                "stock": stock,
                "location": location,
                "bed_bath_ratio": bedrooms / bathrooms,
            }
        ],
        columns=FEATURES,
    )
    return float(bundle["model"].predict(sample)[0])


if __name__ == "__main__":
    print(f"Predicted price: KES {predict_sample_property():,.0f}")
