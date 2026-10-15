from pathlib import Path

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "property_price_model.joblib"
NUMERIC_FEATURES = [
    "bedrooms",
    "bathrooms",
    "square_meters",
    "rating",
    "stock",
    "bed_bath_ratio",
]
FEATURES = NUMERIC_FEATURES + ["location"]


def build_feature_frame(properties):
    data = properties.copy()
    if "bed_bath_ratio" not in data:
        data["bed_bath_ratio"] = data["bedrooms"] / data["bathrooms"]
    return data[FEATURES]


def build_model(regressor=None):
    preprocess = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            ("location", OneHotEncoder(handle_unknown="ignore"), ["location"]),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocess", preprocess),
            (
                "regressor",
                clone(regressor)
                if regressor is not None
                else RandomForestRegressor(
                    n_estimators=200, min_samples_leaf=2, random_state=42, n_jobs=-1
                ),
            ),
        ]
    )


def train_property_model(
    data_path=None,
    model_path=MODEL_PATH,
    regressor=None,
):
    data_path = Path(data_path or PROJECT_ROOT / "data" / "properties_clean.csv")
    if not data_path.is_file():
        raise FileNotFoundError(
            f"Clean property data not found at {data_path}. Import and process listings first."
        )
    data = pd.read_csv(data_path)
    if len(data) < 5:
        raise ValueError("At least five valid listings are required to train the model.")

    features = build_feature_frame(data)
    target = pd.to_numeric(data["price"], errors="coerce")
    valid = target.notna() & features.notna().all(axis=1)
    features = features.loc[valid]
    target = target.loc[valid]
    if len(features) < 5:
        raise ValueError("At least five complete property listings are required to train.")

    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.2, random_state=42
    )
    evaluation_model = build_model(regressor)
    evaluation_model.fit(x_train, y_train)
    mae = float(mean_absolute_error(y_test, evaluation_model.predict(x_test)))

    model = build_model(regressor)
    model.fit(features, target)
    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "mae": mae, "training_rows": len(features)}, model_path)
    return {"mae": mae, "training_rows": len(features), "model_path": str(model_path)}


if __name__ == "__main__":
    result = train_property_model()
    print(
        f"Trained on {result['training_rows']} listings. "
        f"Holdout MAE: KES {result['mae']:,.0f}. Saved to {result['model_path']}"
    )
