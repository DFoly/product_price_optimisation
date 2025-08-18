# COMMAND ----------
import pandas as pd
import yaml
from loguru import logger
# from pyspark.sql import SparkSession
from databricks.connect import DatabricksSession

from pricing.config import ProjectConfig
from pricing.data_processor import DataProcessor

config = ProjectConfig.from_yaml(config_path="../project_config.yml", env="dev")

logger.info("Configuration loaded:")
logger.info(yaml.dump(config, default_flow_style=False))

# COMMAND ----------
spark = DatabricksSession.builder.getOrCreate() # allows to switch between local and remote Spark clusters
data_processor = DataProcessor(config, spark)

# COMMAND ----------
print(data_processor.config)

# COMMAND ----------
# self = data_processor
# df_sales = self.spark.table(f"{self.config.catalog_name}.{self.config.schema_name}.sales_train_evaluation").toPandas()
# df_calendar = self.spark.table(f"{self.config.catalog_name}.{self.config.schema_name}.calendar").toPandas()
# df_prices = self.spark.table(f"{self.config.catalog_name}.{self.config.schema_name}.sell_prices").toPandas()

# all_d_cols = [col for col in df_sales.columns if col.startswith('d_')]
# sales_d_cols = all_d_cols[-self.num_days_train:]

# COMMAND ----------
df = data_processor.preprocess()
# COMMAND ----------
display(df.limit(10))
# COMMAND ----------
