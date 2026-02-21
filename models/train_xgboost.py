import pandas as pd
from sqlalchemy import create_engine
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error
import xgboost as xgb
import joblib

engine = create_engine(
    "postgresql://postgres:yourpassword@localhost:5432/market_intelligence")
df = pd.read_sql("SELECT * FROM properties_clean", engine)

X = df.drop(["price", "property_id", "title", "listing_date"], axis=1)
y = df["price"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)
model = xgb.XGBRegressor(objective="reg:squarederror")
params = {
    "n_estimators": [100, 200],
    "max_depth": [3, 5],
    "learning_rate": [0.01, 0.1]
}

best_model = grid.best_estimator_
preds = best_model.predict(X_test)
mae = mean_absolute_error(y_test, preds)

print("Best Params:", grid.best_params_)
print("XGBOOST MAE:", mae)

joblib.dump(best_model, "models/xgboost_property_price_model.pkl")
