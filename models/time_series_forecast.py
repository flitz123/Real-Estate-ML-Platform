import pandas as pd
from sqlalchemy import create_engine
from sklearn.linear_model import LinearRegression
import numpy as np
import matplotlib.pyplot as plt

engine = create_engine(
    "postgresql://postgres:postdb12@localhost:5432/market_intelligence")
df = pd.read_sql("SELECT listing_date, price FROM properties_clean", engine)

df["listing_date"] = pd.to_datetime(df["listing_date"])
df = df.sort_values("listing_date")
df["days"] = (df["listing_date"] - df["listing_date"].min()).dt.days

X = df[["days"]]
y = df["price"]

model = LinearRegression()
model.fit(X, y)

future_days = np.arange(df["days"].max(), df["days"].max()+30).reshape(-1, 1)
forecast = model.predict(future_days)

plt.plot(df["listing_date"], y, label="Historical")
plt.plot(pd.date_range(df["listing_date"].max(),
         periods=30), forecast, label="Forecast")
plt.legend()
plt.title("Simulated Property Price Forecast")
plt.show()
