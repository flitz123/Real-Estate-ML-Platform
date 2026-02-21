import requests
import pandas as pd
import time
import random


class PropertyAPIScraper:
    def __init__(self):
        self.base_url = "https://dummyjson.com/products"
        self.limit = 10
        self.locations = ["Nairobi", "Eldoret", "Mombasa", "Kisumu", "Nakuru"]

    def fetch_page(self, skip):
        params = {"limit": self.limit, "skip": skip}
        response = requests.get(self.base_url, params=params)
        response.raise_for_status()
        return response.json()

    def stimulate_property_features(self, product):
        return {
            "property_id": product["id"],
            "title": product["title"],
            # Simulate property price in KES
            "price": product["price"] * random.randint(8800, 15000),
            "bedrooms": random.randint(1, 6),
            "bathrooms": random.randint(1, 4),
            "square_metres": random.randint(40, 400),
            "locations": random.choice(self.locations),
            "rating": product["rating"],
            "stock": product["stock"]
        }

    def scrape(self):
        all_propeties = []
        skip = 0

        while True:
            data = self.fetch_page(skip)
            if not data["products"]:
                break

            for product in data["products"]:
                all_propeties.append(self.stimulate_property_features(product))

                skip += self.limit
                time.sleep(1)

        df = pd.DataFrame(all_propeties)
        df.to_csv("data/raw/properties_raw.csv", index=False)
        print("Property data created")
        return df


if __name__ == "__main__":
    scraper = PropertyAPIScraper()
    scraper.scrape()
