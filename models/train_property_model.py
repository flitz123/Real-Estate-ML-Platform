from models.model_pipeline import train_property_model


if __name__ == "__main__":
    result = train_property_model()
    print(
        f"Trained on {result['training_rows']} listings. "
        f"Holdout MAE: KES {result['mae']:,.0f}. Saved to {result['model_path']}"
    )
