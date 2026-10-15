from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "properties_raw.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "properties_clean.csv"

REQUIRED_COLUMNS = {
    "property_id",
    "title",
    "price",
    "bedrooms",
    "bathrooms",
    "square_meters",
    "location",
    "rating",
    "stock",
}


def transform_property_data(
    input_path=RAW_DATA_PATH,
    output_path=PROCESSED_DATA_PATH,
):
    input_path = Path(input_path)
    if not input_path.is_file():
        raise FileNotFoundError(
            f"Property data not found at {input_path}. Import listings first."
        )

    df = pd.read_csv(input_path)
    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError(f"Property data is missing columns: {', '.join(sorted(missing))}")
    if df.empty:
        raise ValueError("Property data contains no listings.")

    df = df.drop_duplicates(subset=["property_id"]).copy()
    numeric_columns = [
        "price",
        "bedrooms",
        "bathrooms",
        "square_meters",
        "rating",
        "stock",
    ]
    for column in numeric_columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")
    df = df.dropna(subset=numeric_columns + ["location"])
    df = df[
        (df["price"] > 0)
        & (df["bedrooms"] > 0)
        & (df["bathrooms"] > 0)
        & (df["square_meters"] > 0)
    ].copy()
    if df.empty:
        raise ValueError("No valid property listings remain after cleaning.")

    df["price_per_sqm"] = df["price"] / df["square_meters"]
    df["bed_bath_ratio"] = df["bedrooms"] / df["bathrooms"]
    df["listing_date"] = pd.date_range(
        end=pd.Timestamp.now().normalize(), periods=len(df)
    )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    return df


if __name__ == "__main__":
    cleaned = transform_property_data()
    print(f"Cleaned {len(cleaned)} property listings into {PROCESSED_DATA_PATH}")
