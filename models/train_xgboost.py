from xgboost import XGBRegressor

from models.model_pipeline import train_property_model


if __name__ == "__main__":
    estimator = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        random_state=42,
        n_jobs=-1,
    )
    result = train_property_model(regressor=estimator)
    print(
        f"Trained XGBoost on {result['training_rows']} listings. "
        f"Holdout MAE: KES {result['mae']:,.0f}. Saved to {result['model_path']}"
    )
