import pandas as pd
from sqlalchemy import create_engine
from sklearn.cluster import KMeans
import joblib

engine = create_engine(
    "postgresql://postgres:yourpassword@localhost:5432/market_intelligence")
df = pd.read_sql("SELECT * FROM properties_clean", engine)

features = df[["price", "square_metres"]]

kmeans = KMeans(n_clusters=3, random_state=42)
df["location_cluster"] = kmeans.fit_predict(features)

df.to_sql("properties_clustered", engine, if_exists="replace", index=False)
joblib.dump(kmeans, "models/kmeans_model.pkl")

print("Location clustering completed and model saved.")
