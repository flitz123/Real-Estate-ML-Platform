from pathlib import Path
import random
import time

import pandas as pd
import requests


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = PROJECT_ROOT / "data" / "properties_raw.csv"


class PropertyAPIScraper:
    """Build a clearly simulated Kenyan property dataset from DummyJSON products."""

    PRICE_PER_SQUARE_METER = {
        "Nairobi": 125_000,
        "Mombasa": 82_000,
        "Nakuru": 58_000,
        "Kisumu": 55_000,
        "Eldoret": 48_000,
    }
    PROPERTY_TYPES = (
        "Garden apartment",
        "Urban townhouse",
        "Family home",
        "Modern flat",
        "Courtyard villa",
        "Hill-view bungalow",
    )

    def __init__(self, limit=100, request_timeout=15):
        self.base_url = "https://dummyjson.com/products"
        self.limit = limit
        self.request_timeout = request_timeout
        self.locations = ["Nairobi", "Eldoret", "Mombasa", "Kisumu", "Nakuru"]

    def fetch_page(self, skip):
        response = requests.get(
            self.base_url,
            params={"limit": self.limit, "skip": skip},
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data.get("products"), list):
            raise ValueError("The product API returned an invalid response.")
        return data

    def simulate_property_features(self, product):
        property_id = int(product["id"])
        rng = random.Random(property_id)
        location = rng.choice(self.locations)
        square_meters = rng.randint(40, 400)
        bedrooms = max(1, min(6, round(square_meters / 65)))
        bathrooms = max(1, min(4, round(bedrooms * 0.65)))
        price_per_square_meter = self.PRICE_PER_SQUARE_METER[location]
        return {
            "property_id": property_id,
            "title": f"{rng.choice(self.PROPERTY_TYPES)} · {location}",
            "price": round(
                square_meters
                * price_per_square_meter
                * (1 + 0.035 * (bedrooms - 2))
                * rng.uniform(0.92, 1.08)
            ),
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "square_meters": square_meters,
            "location": location,
            "rating": float(product["rating"]),
            "stock": int(product["stock"]),
        }

    def scrape(self, output_path=RAW_DATA_PATH, delay=0):
        properties = []
        skip = 0

        while True:
            data = self.fetch_page(skip)
            products = data["products"]
            if not products:
                break

            properties.extend(self.simulate_property_features(item) for item in products)
            skip += len(products)
            total = data.get("total")
            if total is not None and skip >= total:
                break
            if delay:
                time.sleep(delay)

        if not properties:
            raise ValueError("The product API returned no properties.")

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(properties).to_csv(output_path, index=False)
        return pd.DataFrame(properties)


if __name__ == "__main__":
    result = PropertyAPIScraper().scrape()
    print(f"Created {len(result)} simulated property listings at {RAW_DATA_PATH}")
