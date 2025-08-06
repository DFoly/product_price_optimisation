"""Data preprocessing module for Marvel characters."""

import time

import numpy as np
import pandas as pd
from pyspark.sql import SparkSession
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from sklearn.model_selection import train_test_split
from pricing.config import ProjectConfig # handles project configuration


class DataProcessor:
    """
    """

    def __init__(self, pandas_df: pd.DataFrame, config: ProjectConfig, spark: SparkSession) -> None:
        self.df = pandas_df  # Store the DataFrame as self.df
        self.config = config  # Store the configuration
        self.spark = spark
        self.num_days_train = self.config.num_days_train
        self.num_items_subset = self.config.num_items_subset
        self.num_stores_subset = self.config.num_stores_subset

    def preprocess(self) -> None:
        """Loads M5 datasets, merges them, and applies initial filtering and feature engineering.
        """
        cat_features = self.config.cat_features
        num_features = self.config.num_features
        target = self.config.target

        # read tables in from catalog
        df_sales = self.spark.table(f"{self.config.catalog_name}.{self.config.schema_name}.sales")
        df_calendar = self.spark.table(f"{self.config.catalog_name}.{self.config.schema_name}.calendar")
        df_prices = self.spark.table(f"{self.config.catalog_name}.{self.config.schema_name}.sell_prices")

        # Determine the sales columns to load based on num_days_train
        all_d_cols = [col for col in df_sales.columns if col.startswith('d_')]
        sales_d_cols = all_d_cols[-self.num_days_train:]
        
        # convert to long format
        id_cols = ['id', 'item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']
        df_sales_melted = df_sales[id_cols + sales_d_cols].melt(
            id_vars=id_cols,
            var_name='d',
            value_name='demand' # Renamed to 'demand' as per user's snippet
        )
        df_sales_melted['d'] = df_sales_melted['d'].str[2:].astype("int64")
        df_calendar['d'] = df_calendar['d'].str[2:].astype("int64")
        df_sales_melted['demand'] = df_sales_melted['demand'].astype("float32")
        print(f"Sales data melted. Total rows: {len(df_sales_melted)}")

        # Merge with calendar data
        df_merged = pd.merge(df_sales_melted, df_calendar, on='d', how='left')
        df_merged['date'] = pd.to_datetime(df_merged['date'])
        print("Merged with calendar data.")

        # Merge with sell_prices data
        df_merged = pd.merge(df_merged, df_prices, on=['store_id', 'item_id', 'wm_yr_wk'], how='left')
        df_merged = df_merged.drop(["wm_yr_wk"], axis=1) # Drop wm_yr_wk as per user's snippet
        print("Merged with sell_prices data.")

        # Filter to a manageable subset for demonstration
        selected_items = df_sales['item_id'].unique()[:self.num_items_subset]
        selected_stores = df_sales['store_id'].unique()[:self.num_stores_subset]

        df_filtered = df_merged[
            (df_merged['item_id'].isin(selected_items)) &
            (df_merged['store_id'].isin(selected_stores))
        ].copy()
        
        # Fill missing sell_price values (as done in previous code)
        df_filtered['sell_price'] = df_filtered.groupby(['id'])['sell_price'].transform(lambda x: x.ffill().bfill())
        df_filtered['sell_price'].fillna(df_filtered['sell_price'].mean(), inplace=True)
        print("Filled missing sell_price values.")

        # Calculate `days_since_first_sale`
        df_filtered['first_sale_date'] = df_filtered.groupby('id')['date'].transform('min')
        df_filtered['days_since_first_sale'] = (df_filtered['date'] - df_filtered['first_sale_date']).dt.days
        print("Calculated days_since_first_sale.")

        print(f"\nFiltered dataset to {num_items_subset} items and {num_stores_subset} stores.")
        print(f"Total rows in filtered data: {len(df_filtered)}")
        print("Sample Filtered Data Head:")
        print(df_filtered.head())
        
        return df_filtered, df_calendar

      


    def split_data(self, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the DataFrame (self.df) into training and test sets.

        :param test_size: The proportion of the dataset to include in the test split.
        :param random_state: Controls the shuffling applied to the data before applying the split.
        :return: A tuple containing the training and test DataFrames.
        """
        train_set, test_set = train_test_split(self.df, test_size=test_size, random_state=random_state)
        return train_set, test_set

    def save_to_catalog(self, train_set: pd.DataFrame, test_set: pd.DataFrame) -> None:
        """
         Save the train and test sets into Databricks tables.

        :param train_set: The training DataFrame to be saved.
        :param test_set: The test DataFrame to be saved.
        """
        train_set_with_timestamp = self.spark.createDataFrame(train_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        test_set_with_timestamp = self.spark.createDataFrame(test_set).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        train_set_with_timestamp.write.mode("overwrite").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.train_set"
        )

        test_set_with_timestamp.write.mode("overwrite").saveAsTable(
            f"{self.config.catalog_name}.{self.config.schema_name}.test_set"
        )

    def enable_change_data_feed(self) -> None:
        """Enable Change Data Feed for train and test set tables.

        This method alters the tables to enable Change Data Feed functionality.
        """
        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.train_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )

        self.spark.sql(
            f"ALTER TABLE {self.config.catalog_name}.{self.config.schema_name}.test_set "
            "SET TBLPROPERTIES (delta.enableChangeDataFeed = true);"
        )


