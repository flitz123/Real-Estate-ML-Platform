import os
import re

import pandas as pd
from sqlalchemy import create_engine


_TABLE_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


class DatabaseManager:
    def __init__(self, database_url=None):
        url = database_url or os.getenv("DATABASE_URL")
        if not url:
            raise ValueError(
                "Set DATABASE_URL to a SQLAlchemy connection URL before using DatabaseManager."
            )
        self.engine = create_engine(url, pool_pre_ping=True)

    @staticmethod
    def _validate_table_name(table_name):
        if not _TABLE_NAME.fullmatch(table_name):
            raise ValueError(f"Invalid table name: {table_name!r}")
        return table_name

    def insert_database(self, df, table_name):
        table_name = self._validate_table_name(table_name)
        df.to_sql(table_name, self.engine, if_exists="replace", index=False)
        print(f"Data inserted into {table_name} table successfully.")

    def read_table(self, table_name):
        table_name = self._validate_table_name(table_name)
        return pd.read_sql_table(table_name, self.engine)
