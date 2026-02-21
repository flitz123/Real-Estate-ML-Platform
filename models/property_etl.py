import pandas as pd
from database.db import DatabaseManager


def transform_property_data():
    db = DatabaseManager()
    df = db.read_table("properties_raw")

    df["price_per_sqm"] = df["price"] / df["square_metres"]
    df["bed_bath_ratio"] = df["bedrooms"] / df["bathrooms"]

    df = pd.get_dummies(df, columns=["locations"], drop_first=True)
    df["listing_date"] = pd.date_range(
        end=pd.Timestamp.today(), periods=len(df))

    db.insert_database(df, "properties_clean")
    print("ETL and transformation completed")
    return df
