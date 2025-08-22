# This would be part of a daily data pipeline that takes in new data creates a features and updates our feature store
import time
import sys
import numpy as np
import pandas as pd
from databricks.connect import DatabricksSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import current_timestamp, to_utc_timestamp
from src.pricing.config import ProjectConfig
from databricks.feature_engineering import FeatureEngineeringClient


class DataProcessor:
    """
    DataProcessor is responsible for preprocessing the data for training and testing.
    """
    def __init__(self, config: ProjectConfig, spark: DatabricksSession) -> None:
        self.config = config  # Store the configuration
        self.spark = spark
        self.num_days_train = self.config.parameters['num_days_train']
        self.num_items_subset = self.config.parameters['num_items_subset']
        self.num_stores_subset = self.config.parameters['num_stores_subset']
        self.split_date = pd.to_datetime("2016-04-25")  # Default split date
        self.testing = self.config.parameters.get('testing', False)


    def preprocess(self):
        """Loads M5 datasets, merges them, and applies initial filtering and feature engineering using PySpark."""

        # Read tables from catalog
        df_sales = self.spark.table(f"{self.config.catalog_name}.{self.config.schema_name}.sales_train_evaluation")
        df_calendar = self.spark.table(f"{self.config.catalog_name}.{self.config.schema_name}.calendar")
        df_prices = self.spark.table(f"{self.config.catalog_name}.{self.config.schema_name}.sell_prices")

        # Determine the sales columns to load based on num_days_train
        all_d_cols = [col for col in df_sales.columns if col.startswith('d_')]
        sales_d_cols = all_d_cols[-self.num_days_train:]
        id_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']

        df_sales_selected = df_sales.select(id_cols + sales_d_cols)
        df_sales_melted = (
            df_sales_selected
            .melt(
                ids=id_cols, 
                values=sales_d_cols, 
                variableColumnName='d', 
                valueColumnName='demand'
            )
            .withColumn("d", F.expr("int(substr(d, 3, length(d)))"))
            .withColumn("demand", F.col("demand").cast("float"))
        )

        # Merge with calendar and sell_prices
        df_merged = (
            df_sales_melted
            .join(df_calendar, on="d", how="left")
            .withColumn("date", F.to_date("date"))
            .join(df_prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
            .drop("wm_yr_wk")
        )

        if self.testing:
            # Filter to a manageable subset for testing
            print(f"\nFiltered dataset to {self.num_items_subset} items and {self.num_stores_subset} stores.")
            selected_items = [row.item_id for row in df_sales.select("item_id").distinct().limit(self.num_items_subset).collect()]
            selected_stores = [row.store_id for row in df_sales.select("store_id").distinct().limit(self.num_stores_subset).collect()]
            df_filtered = df_merged.filter(
                (F.col("item_id").isin(selected_items)) &
                (F.col("store_id").isin(selected_stores))
            )
        else:
            df_filtered = df_merged

        # create id column: concat(item_id, store_id)
        # create primary key: pk_id = concat(item_id, store_id, date)
        # ITEM STORE IDENTIFIER: HOBBIES_1_001_CA_1
        df_filtered = (
            df_filtered
            .withColumn("id_pk", F.concat(F.col("item_id"), F.lit("_"), F.col("store_id"), F.lit("_"), F.col('date')))
            .withColumn("id", F.concat(F.col("item_id"), F.lit("_"), F.col("store_id")))
        )

        # get average by window and then coalesce: average up to current row
        window_spec = Window.partitionBy("item_id").orderBy("date").rowsBetween(Window.unboundedPreceding, Window.currentRow-1)
        df_filtered = df_filtered.withColumn("sell_price", F.coalesce("sell_price", F.mean("sell_price").over(window_spec)))

        # Calculate days_since_first_sale
        window_first_sale = Window.partitionBy("id")
        df_filtered = df_filtered.withColumn(
            "first_sale_date",
            F.min("date").over(window_first_sale)
        )
        df_filtered = df_filtered.withColumn(
            "days_since_first_sale",
            F.datediff(F.col("date"), F.col("first_sale_date"))
        )
        print(f"Total rows in filtered data: {df_filtered.count()}")

        self.df = df_filtered
        return self.df


    def split_data(self, split_date: str = None, test_size: float = 0.2, random_state: int = 42) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Split the DataFrame (self.df) into training and test sets.
           split on time dimension, not random sample: d
        """
        if not split_date:
            split_date = self.split_date

        train_set = self.df[self.df['d'] < split_date]
        test_set = self.df[self.df['d'] >= split_date]
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


class FeatureProducer:
    """
    Generate features and save them in databricks feature store
    """
    def __init__(self, spark: DatabricksSession):
        self.spark=spark
        self.lags = [7, 14, 28]
        self.windows = [7]
        self.fe = FeatureEngineeringClient()

    def label_encoder():
        """
            Handle the categorical data
        """
        pass


    def generate_features_spark(self, df_input):
        """
        """
        print("Starting Spark-based feature generation...")
        
        # 1. Generate time and event features
        df_with_time_features = df_input.withColumn(
            "dayofweek", F.dayofweek("date")
        ).withColumn(
            "month", F.month("date")
        ).withColumn(
            "year", F.year("date")
        ).withColumn(
            "week", F.weekofyear("date")
        ).withColumn(
            "dayofyear", F.dayofyear("date")
        ).withColumn(
            "is_event", F.when(F.col("event_name_1").isNotNull(), 1).otherwise(0)
        )
        
        # 2. Generate lagged and rolling features
        # Sort the data by id and date for correct windowing
        df_sorted = df_with_time_features.sort(F.col("id"), F.col("date"))
        
        lag_cols = []
        roll_cols = []

        for lag in self.lags:
            window_spec_lag = Window.partitionBy("id").orderBy("date")
            lag_col_name = f'lag_t{lag}'
            df_lagged = df_sorted.withColumn(
                lag_col_name, F.lag("demand", lag).over(window_spec_lag).cast("float")
            )
            lag_cols.append(lag_col_name)
            # Add rolling means for the newly created lag feature
            for w in self.windows:
                roll_col_name = f'rolling_mean_lag{lag}_w{w}'
                window_spec_roll = Window.partitionBy("id").orderBy("date").rowsBetween(-w, -1)
                df_lagged = df_lagged.withColumn(
                    roll_col_name, F.avg(lag_col_name).over(window_spec_roll).cast("float")
                )
                roll_cols.append(roll_col_name)
            df_sorted = df_lagged

        # add summary stats: running average etc
        item_window_spec =  Window.partitionBy("item_id").orderBy("date").rowsBetween(Window.unboundedPreceding, 0)
        store_window_spec =  Window.partitionBy("store_id").orderBy("date").rowsBetween(Window.unboundedPreceding, 0)

        df_sorted = (
            df_sorted
             .withColumn('item_running_avg', F.avg('demand').over(item_window_spec))
             .withColumn('store_running_avg', F.avg('demand').over(store_window_spec))
        )

        # Add timestamps etc.
        df_final = df_sorted.withColumn(
            "event_timestamp", F.to_timestamp("date")
        ).withColumn(
            "update_timestamp_utc", to_utc_timestamp(current_timestamp(), "UTC")
        )

        # Remove rows with nulls in any of the new lag/rolling columns
        cols_to_check_nulls = lag_cols + roll_cols
        df_final_cleaned = df_final.dropna(subset=cols_to_check_nulls)
        return df_final_cleaned


    def publish_features(self, df_features, table_name):
        """
        Publishes the generated features to a Databricks Feature Store table.
        https://docs.databricks.com/aws/en/machine-learning/feature-store/time-series
        """
        print(f"Publishing features to table: {table_name}")
        try:
            # Try to get the table; if it exists this succeeds
            # what is the default behaviour if the keys already exisit? i.e. will it be an upsert?
            self.fe.get_table(name=table_name)
            self.fe.write_table(
                name=table_name,
                df=df_features,
                mode="overwrite"
            )
            print("Table existed. Overwritten.")
        except ValueError:
            # Table does not exist: Create
            self.fe.create_table(
                name=table_name,
                df=df_features,
                primary_keys=["id", "date"],
                timeseries_columns=['date'],
                description="Time-series and sales features for M5 forecasting.",
            )
