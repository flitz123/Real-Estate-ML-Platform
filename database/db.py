from sqlalchemy import create_engine
import pandas as pd


class DatabaseManager:
    def __init__(self):
        self.engine = create_engine(
            "postgresql://postgres:postdb12@localhost:5432/market_intelligence"
        )

    def insert_database(self, df, table_name):
        df.to_sql(table_name, self.engine, if_exists="replace", index=False)
        print(f"Data inserted into {table_name} table successfully.")

    def read_table(self, table_name):
        return pd.read_sql(f"SELECT * FROM {table_name}", self.engine)
