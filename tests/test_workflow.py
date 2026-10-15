import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import pandas as pd
from fastapi import HTTPException

from api import main
from models.location_clustering import cluster_locations
from models.model_pipeline import FEATURES, train_property_model
from models.property_etl import transform_property_data
from models.time_series_forecast import forecast_prices
from scraper.property_api_scraper import PropertyAPIScraper


def sample_properties(count=12):
    locations = ["Nairobi", "Mombasa", "Kisumu", "Nakuru"]
    return pd.DataFrame(
        [
            {
                "property_id": index + 1,
                "title": f"Home {index + 1}",
                "price": 5_000_000 + index * 250_000,
                "bedrooms": 1 + index % 5,
                "bathrooms": 1 + index % 3,
                "square_meters": 60 + index * 8,
                "location": locations[index % len(locations)],
                "rating": 3.5 + (index % 4) * 0.3,
                "stock": 5 + index,
            }
            for index in range(count)
        ]
    )


class WorkflowTests(unittest.TestCase):
    def test_bundled_demo_data_and_model_are_ready(self):
        health = main.health()
        market = main.market()
        self.assertTrue(health["model_ready"])
        self.assertTrue(health["market_data_ready"])
        self.assertGreater(market["total_listings"], 0)
        self.assertEqual(len(market["listings"]), market["total_listings"])
        prediction = main.predict(
            main.PredictionRequest(
                location="Nairobi",
                bedrooms=3,
                bathrooms=2,
                square_meters=120,
                rating=4.2,
                stock=12,
            )
        )
        self.assertGreater(prediction["predicted_price"], 0)

    def test_scraper_paginates_using_products_returned(self):
        scraper = PropertyAPIScraper(limit=2)
        pages = [
            {"products": [{"id": 1, "title": "First", "price": 10, "rating": 4, "stock": 5}], "total": 3},
            {"products": [{"id": 2, "title": "Second", "price": 20, "rating": 4, "stock": 5}], "total": 3},
            {"products": [{"id": 3, "title": "Third", "price": 30, "rating": 4, "stock": 5}], "total": 3},
        ]
        requested_offsets = []

        def fetch_page(skip):
            requested_offsets.append(skip)
            return pages[len(requested_offsets) - 1]

        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "properties.csv"
            with patch.object(scraper, "fetch_page", side_effect=fetch_page):
                properties = scraper.scrape(output, delay=0)
            self.assertEqual(requested_offsets, [0, 1, 2])
            self.assertEqual(len(properties), 3)
            self.assertEqual(len(pd.read_csv(output)), 3)
            self.assertEqual(
                properties.iloc[0]["price"],
                scraper.simulate_property_features(pages[0]["products"][0])["price"],
            )

    def test_etl_cleans_and_derives_features(self):
        with tempfile.TemporaryDirectory() as directory:
            raw_path = Path(directory) / "raw.csv"
            clean_path = Path(directory) / "processed" / "clean.csv"
            sample_properties().to_csv(raw_path, index=False)
            cleaned = transform_property_data(raw_path, clean_path)
            self.assertTrue(clean_path.is_file())
            self.assertEqual(len(cleaned), 12)
            self.assertAlmostEqual(
                cleaned.iloc[0]["price_per_sqm"],
                cleaned.iloc[0]["price"] / cleaned.iloc[0]["square_meters"],
            )
            self.assertAlmostEqual(
                cleaned.iloc[0]["bed_bath_ratio"],
                cleaned.iloc[0]["bedrooms"] / cleaned.iloc[0]["bathrooms"],
            )

    def test_trained_model_predicts_and_api_serves_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw_path = root / "properties.csv"
            clean_path = root / "clean.csv"
            model_path = root / "model.joblib"
            data = sample_properties()
            data.to_csv(raw_path, index=False)
            transform_property_data(raw_path, clean_path)
            training = train_property_model(clean_path, model_path)
            self.assertEqual(training["training_rows"], len(data))
            self.assertTrue(model_path.is_file())

            with (
                patch.object(main, "MODEL_PATH", model_path),
                patch.object(main, "RAW_DATA_PATH", raw_path),
            ):
                self.assertTrue(main.health()["model_ready"])
                market = main.market()
                self.assertEqual(market["total_listings"], len(data))
                self.assertEqual(len(market["listings"]), len(data))
                self.assertNotIn("price_per_sqm", FEATURES)
                prediction = main.predict(
                    main.PredictionRequest(
                        location="Nairobi",
                        bedrooms=3,
                        bathrooms=2,
                        square_meters=120,
                        rating=4.2,
                        stock=12,
                    )
                )
                self.assertGreater(prediction["predicted_price"], 0)
                self.assertEqual(Path(main.home().path), main.WEB_DIR / "index.html")

            forecast = forecast_prices(
                clean_path, root / "forecast.csv", days=7
            )
            self.assertEqual(len(forecast), 7)
            self.assertTrue((root / "forecast.csv").is_file())
            clustered = cluster_locations(
                clean_path, root / "clustered.csv", root / "cluster-model.joblib"
            )
            self.assertEqual(clustered["location_cluster"].nunique(), 3)

    def test_prediction_reports_missing_model(self):
        with tempfile.TemporaryDirectory() as directory:
            missing_model = Path(directory) / "missing.joblib"
            with patch.object(main, "MODEL_PATH", missing_model):
                with self.assertRaises(HTTPException) as raised:
                    main.predict(
                        main.PredictionRequest(
                            location="Nairobi",
                            bedrooms=3,
                            bathrooms=2,
                            square_meters=120,
                            rating=4.2,
                            stock=12,
                        )
                    )
            self.assertEqual(raised.exception.status_code, 503)
            self.assertIn("Import demo listings", raised.exception.detail)

    def test_market_refresh_is_disabled_for_serverless_deployments(self):
        with patch.object(main, "SYNC_ENABLED", False):
            self.assertFalse(main.health()["sync_enabled"])
            with self.assertRaises(HTTPException) as raised:
                main.sync_market()
        self.assertEqual(raised.exception.status_code, 503)
        self.assertIn("temporary", raised.exception.detail)


if __name__ == "__main__":
    unittest.main()
