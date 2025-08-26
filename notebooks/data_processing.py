# Databricks notebook source
# %pip install -e ..
# %restart_python

# COMMAND ----------

# fix path for imports
# from pathlib import Path
# import sys
# sys.path.append(str(Path.cwd().parent / 'src'))

# COMMAND ----------

import yaml
import importlib
import pandas as pd
from loguru import logger
from databricks.connect import DatabricksSession
import pyspark.sql.functions as F

import mlflow
import mlflow.tracking._model_registry.utils
import src.pricing
from src.pricing.config import ProjectConfig
from src.pricing.data_processor import DataProcessor, FeatureProducer

# Workaround to set the registry URI manually: fixes issue with feature engineering
# mlflow.tracking._model_registry.utils._get_registry_uri_from_spark_session = lambda: "databricks-uc"
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup

config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev") # Ensure 'num_features' is defined in the YAML

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

importlib.reload(src.pricing.data_processor)

spark = DatabricksSession.builder.getOrCreate() # allows to switch between local and remote Spark clusters

%load_ext autoreload
%autoreload 2

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
# # Preprocess the data
df = data_processor.preprocess()

table_name = f"{data_processor.config.catalog_name}.{data_processor.config.schema_name}.price_features"
print(table_name)

# COMMAND ----------

# %sql drop table dev.price_optimisation.price_features

# COMMAND ----------

fe_instance = FeatureProducer(spark)
df_features = fe_instance.generate_features_spark(df)

fe_instance.publish_features(df_features, table_name)
display(df_features.limit(10))

# COMMAND ----------

print(df.count()), print(df_features.count())

# COMMAND ----------

display(df_features.agg(F.countDistinct('id')))

# COMMAND ----------

# MAGIC %sql select count(distinct id) from dev.price_optimisation.price_features

# COMMAND ----------

# MAGIC %sql select * from dev.price_optimisation.price_features order by id, date asc limit 100

# COMMAND ----------

# %sql drop table dev.price_optimisation.price_features

# COMMAND ----------

fe = FeatureEngineeringClient()

feature_names = ['wday', 'month', 'year', 'dayofweek', 'dayofyear', 'week', 'snap_CA', 'snap_TX', 'snap_WI', 
                 'sell_price', 'days_since_first_sale', 'is_event', 'lag_t7', 'rolling_mean_lag7_w7', 'lag_t28', 'rolling_mean_lag28_w7', 'item_running_avg', 'store_running_avg']

# need to pass to feature store create training set
# this could be a separate table which stores the demand per unique id and date
df_features = spark.sql(f"select * from {table_name}")
training_df = df_features.select('id', 'date', 'demand').drop_duplicates()

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
training_df_pd = training_df.toPandas()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Baseline Demand Model
# MAGIC - estimate baseline demand without sell price

# COMMAND ----------

import mlflow

with mlflow.start_run(run_name="baseline_demand_lgbm"):
    mlflow.log_params({"model": "LGBMRegressor", "objective": "poisson", "cv_folds": 4})
    from lightgbm import LGBMRegressor
    from sklearn.model_selection import KFold, cross_val_score
    from sklearn.metrics import mean_squared_error, make_scorer
    import numpy as np

    # Feature columns (input): excluding sell_price to estimate base demand
    feature_names = ['wday', 'month', 'year', 'dayofweek', 'dayofyear', 'week', 
                     'snap_CA', 'snap_TX', 'snap_WI', 
                     'days_since_first_sale', 'is_event',
                     'lag_t7', 'rolling_mean_lag7_w7',
                     'lag_t28', 'rolling_mean_lag28_w7',
                     'item_running_avg', 'store_running_avg']

    X = training_df_pd[feature_names]
    y = training_df_pd['demand']

    # LGBMRegressor setup: Poisson objective
    lgbm = LGBMRegressor(objective='poisson')

    # RMSE scorer for cross-validation
    rmse_scorer = make_scorer(mean_squared_error, squared=False)

    # 5-fold cross validation (shuffle for randomness)
    kf = KFold(n_splits=4, shuffle=True, random_state=42)

    cv_rmse_scores = cross_val_score(
        lgbm,
        X,
        y,
        cv=kf,
        scoring=rmse_scorer,
        n_jobs=-1
    )

    mlflow.log_metric("mean_cv_rmse", np.mean(cv_rmse_scores))
    mlflow.log_metric("std_cv_rmse", np.std(cv_rmse_scores))
    for i, score in enumerate(cv_rmse_scores):
        mlflow.log_metric(f"cv_rmse_fold_{i+1}", score)

    print("Cross-validated RMSE scores:", cv_rmse_scores)
    print("Mean CV RMSE:", np.mean(cv_rmse_scores))
    print("Std CV RMSE:", np.std(cv_rmse_scores))

# COMMAND ----------


