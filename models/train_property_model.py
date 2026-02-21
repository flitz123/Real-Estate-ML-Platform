import pandas as pd
import sqlalchemy as create_engine
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

engine = create_engine(
    "postgresql://postgres:yourpassword@localhost:5432/market_intelligence")
df = pd.read_sql("SELECT * FROM properties_clean", engine)

X = df.drop(["price", "property_id", "title", "listing_date"], axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model = RandomForestRegressor(n_estimators=200, random_state=42)
model.fit(X_train, y_train)

predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
print("Random Forest MAE:", mae)

joblib.dump(model, "models/property_price_model.pkl")
