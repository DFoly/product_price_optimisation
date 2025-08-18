# Databricks notebook source

# MAGIC %pip install -e ..
# MAGIC %restart_python

# COMMAND ----------
# from pathlib import Path
# import sys
# sys.path.append(str(Path.cwd().parent / 'src'))

# COMMAND ----------
import yaml
import importlib
import pandas as pd
from loguru import logger
from databricks.connect import DatabricksSession
spark_connect_url = 
spark = DatabricksSession.builder.config("spark.remote", spark_connect_url).getOrCreate() # allows to switch between local and remote Spark clusters

import mlflow
import mlflow.tracking._model_registry.utils
import pricing
from pricing.config import ProjectConfig
from pricing.data_processor import DataProcessor, FeatureProducer

# Workaround to set the registry URI manually: fixes issue with feature engineering
mlflow.tracking._model_registry.utils._get_registry_uri_from_spark_session = lambda: "databricks-uc"
from databricks.feature_engineering import FeatureEngineeringClient

config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

importlib.reload(pricing.data_processor)

# COMMAND ----------
data_processor = DataProcessor(config, spark)

# COMMAND ----------
# Preprocess the data
df = data_processor.preprocess()
df.head()
logger.info(f"Data preprocessing completed.")

# COMMAND ----------
# fe = FeatureProducer(spark)
# df_features = fe.generate_features_spark(df)

# table_name = f"{data_processor.config.catalog_name}.{data_processor.config.schema_name}.price_features"
# fe.publish_features(df_features, table_name, create_table=True)

fe = FeatureEngineeringClient()

# COMMAND ----------

# COMMAND ---------- 

# COMMAND ----------
