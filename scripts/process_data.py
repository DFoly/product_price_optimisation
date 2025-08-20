import argparse
import yaml
from loguru import logger
from pyspark.sql import SparkSession
import pandas as pd

from pricing.config import ProjectConfig
from pricing.data_processor import DataProcessor

parser = argparse.ArgumentParser()
parser.add_argument(
    "--root_path",
    action="store",
    default=None,
    type=str,
    required=True,
)

parser.add_argument(
    "--env",
    action="store",
    default=None,
    type=str,
    required=True,
)


args = parser.parse_args()
config_path = f"{args.root_path}/files/project_config.yml"

config = ProjectConfig.from_yaml(config_path=config_path, env=args.env)

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

spark = SparkSession.builder.getOrCreate()

# Initialize DataProcessor
data_processor = DataProcessor(config, spark)

# Preprocess the data
df = data_processor.preprocess()

# feature engineering and feature store
fe_instance = FeatureProducer(spark)
df_features = fe_instance.generate_features_spark(df)

table_name = f"{data_processor.config.catalog_name}.{data_processor.config.schema_name}.price_features"
fe_instance.publish_features(df_features, table_name, create_table=True)