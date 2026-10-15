from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from models.property_etl import PROCESSED_DATA_PATH


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CLUSTERED_DATA_PATH = PROJECT_ROOT / "data" / "properties_clustered.csv"
CLUSTER_MODEL_PATH = PROJECT_ROOT / "models" / "kmeans_model.joblib"


def cluster_locations(
    data_path=PROCESSED_DATA_PATH,
    output_path=CLUSTERED_DATA_PATH,
    model_path=CLUSTER_MODEL_PATH,
):
    data_path = Path(data_path)
    if not data_path.is_file():
        raise FileNotFoundError(f"Clean property data not found at {data_path}.")
    data = pd.read_csv(data_path)
    if len(data) < 3:
        raise ValueError("At least three listings are required for location clustering.")
    features = data[["price", "square_meters"]].apply(pd.to_numeric, errors="coerce")
    if features.isna().any().any():
        raise ValueError("Price and square meter values must be valid for clustering.")

    model = make_pipeline(
        StandardScaler(),
        KMeans(n_clusters=3, random_state=42, n_init=10),
    )
    data["location_cluster"] = model.fit_predict(features)
    output_path = Path(output_path)
    model_path = Path(model_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    joblib.dump(model, model_path)
    return data


if __name__ == "__main__":
    clustered = cluster_locations()
    print(f"Clustered {len(clustered)} listings into three market groups.")
