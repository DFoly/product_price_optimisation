# Databricks notebook source
# %pip install -e ..
# %restart_python

# COMMAND ----------

# fix path for imports
from pathlib import Path
import sys
sys.path.append(str(Path.cwd().parent / 'src'))

# COMMAND ----------

import yaml
import importlib
import pandas as pd
from loguru import logger
from databricks.connect import DatabricksSession
import pyspark.sql.functions as F

import mlflow
import mlflow.tracking._model_registry.utils
import pricing
from pricing.config import ProjectConfig
from pricing.data_processor import DataProcessor, FeatureProducer

# Workaround to set the registry URI manually: fixes issue with feature engineering
# mlflow.tracking._model_registry.utils._get_registry_uri_from_spark_session = lambda: "databricks-uc"
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

importlib.reload(pricing.data_processor)

spark = DatabricksSession.builder.getOrCreate() # allows to switch between local and remote Spark clusters

# COMMAND ----------

# df_temp = spark.sql("select * from dev.price_optimisation.sales_train_evaluation")
# df_temp = df_temp.withColumn("id", F.concat(F.col("item_id"), F.lit("_"), F.col("store_id"), F.lit("_"), F.col('d_1')))
# display(df_temp.limit(10))

# COMMAND ----------

# df_sales = spark.table("dev.price_optimisation.sales_train_evaluation")
# df_calendar = spark.table("dev.price_optimisation.calendar")
# df_prices = spark.table("dev.price_optimisation.sell_prices")

# num_days_train = 365

# # Determine the sales columns to load based on num_days_train
# all_d_cols = [col for col in df_sales.columns if col.startswith('d_')]
# sales_d_cols = all_d_cols[-num_days_train:]
# id_cols = ['item_id', 'dept_id', 'cat_id', 'store_id', 'state_id']

# df_sales_selected = df_sales.select(id_cols + sales_d_cols)
# df_sales_melted = (
# df_sales_selected
# .melt(
#     ids=id_cols, 
#     values=sales_d_cols, 
#     variableColumnName='d', 
#     valueColumnName='demand'
# )
# .withColumn("d", F.expr("int(substr(d, 3, length(d)))"))
# .withColumn("demand", F.col("demand").cast("float"))
# )

# df_merged = (
#     df_sales_melted
#     .join(df_calendar, on="d", how="left")
#     .withColumn("date", F.to_date("date"))
#     .join(df_prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
#     .drop("wm_yr_wk")
# )

# df_filtered = df_merged.withColumn("id", F.concat(F.col("item_id"), F.lit("_"), F.col("store_id"), F.lit("_"), F.col('d')))

# display(df_filtered)

# COMMAND ----------

data_processor = DataProcessor(config, spark)
# Preprocess the data
df = data_processor.preprocess()

# COMMAND ----------

fe_instance = FeatureProducer(spark)
df_features = fe_instance.generate_features_spark(df)

table_name = f"{data_processor.config.catalog_name}.{data_processor.config.schema_name}.price_features"
fe_instance.publish_features(df_features, table_name, create_table=True)
# display(df_features)

# COMMAND ----------

# MAGIC %sql select * from dev.price_optimisation.price_features where wday IS NULL

# COMMAND ----------

fe = FeatureEngineeringClient()

feature_names = ['wday', 'month', 'year', 'dayofweek', 'dayofyear', 'week', 'snap_CA', 'snap_TX', 'snap_WI', 'sell_price', 'days_since_first_sale', 'is_event', 'lag_t7', 'rolling_mean_lag7_w7', 'lag_t28', 'rolling_mean_lag28_w7', 'item_running_avg', 'store_running_avg']

# need to pass to feature store create training set
# this could be a separate table which stores the demand per unique id and date
training_df = df_features.select('item_store_id', 'date', 'demand').drop_duplicates()

feature_lookups = [
    FeatureLookup(
        table_name=table_name,
        feature_names=feature_names,
        lookup_key="item_store_id",
        timestamp_lookup_key="date"
    )
]

# COMMAND ----------

training_labels = ['demand']

training_set = fe.create_training_set(
    df=training_df,
    feature_lookups=feature_lookups,
    #exclude_columns=["item_store_id"],
    label="demand",
)
training_df = training_set.load_df()

# COMMAND ----------

display(training_df)

# COMMAND ----------


