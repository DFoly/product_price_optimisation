import time
import sys
import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc
from mlflow.models import infer_signature
from mlflow import MlflowClient
from loguru import logger
from databricks.connect import DatabricksSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from pyspark.sql.functions import current_timestamp, to_utc_timestamp, DataFrame
from databricks.feature_engineering import FeatureEngineeringClient
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import lightgbm as lgb


class LGBMModel:
    """
    Base Model for both forecasting and elasticity models.
    This class now uses a pre-computed training DataFrame from the Feature Store.
    """
    def __init__(self, spark, config, params, model_type, training_df: pd.DataFrame, baseline_forecasts_daily_dict: dict):
        self.spark = spark
        self.config = config
        self.params = params
        self.model_type = model_type  # 'forecast' or 'elasticity'
        self.ensemble_models = []
        self.cv_results = {}
        self.categorical_features = ['item_id', 'store_id', 'dept_id', 'cat_id', 'state_id']
        self.catalog_name = self.config.catalog_name
        self.schema_name = self.config.schema_name
        self.experiment_name = self.config.experiment_name_basic + '-' + self.model_type
        self.model_name = f"{self.catalog_name}.{self.schema_name}" + '.' + self.model_type

        self.num_features =  self.config.num_features[self.model_type]
        self.features = self.num_features + self.categorical_features
        
        # Store baseline forecasts for elasticity model data preparation
        self.baseline_forecasts_daily_dict = baseline_forecasts_daily_dict
        
        # Prepare the data differently based on model type
        self.training_df_pandas = self._prepare_data(training_df)

        self.split_date = pd.to_datetime("2016-04-25")
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()


    def _prepare_data(self, training_df: pd.DataFrame) -> pd.DataFrame:
        """
        New method to prepare the training data based on the model type.
        This is where the elasticity target is created.
        """
        if self.model_type == 'elasticity':
            print("Preparing data for elasticity model...")
            # We need to map the baseline forecasts to the training data.
            def get_baseline(row):
                key = (row['item_id'], row['store_id'], row['date'].strftime('%Y-%m-%d'))
                return self.baseline_forecasts_daily_dict.get(key, np.nan)
            
            training_df['baseline_demand'] = training_df.apply(get_baseline, axis=1)

            # Drop rows where we don't have a baseline forecast
            training_df.dropna(subset=['baseline_demand'], inplace=True)
                
            training_df = training_df.sort_values(by=['item_id', 'store_id', 'date'])

            # Baseline Price feature: avoids feature leakage using running average
            training_df['baseline_price'] = (
                training_df
                .groupby(['item_id', 'store_id'])['sell_price']
                .transform(lambda x: x.expanding().mean())
                .shift(1)
            )
                        
            # New feature for the model: price ratio
            training_df['price_ratio'] = training_df['sell_price'] / training_df['baseline_price']

            # Now, add the new features to the model's feature list
            self.features.append('price_ratio')
            self.features.append('baseline_demand')

            # The target remains 'demand' as we are predicting the final value directly
            return training_df
        else:
            # For a baseline forecast model, return the data as is.
            return training_df


    def split_data(self):
        """
            set up the dats for model training below
            we split by date top ensure no feature leakage
        """
        X_train = self.training_df_pandas.query('date < @self.split_date')
        X_test = self.training_df_pandas.query('date >= @self.split_date')
        y_train = X_train['demand']
        y_test = X_test['demand']

        X_train = X_train.drop('demand', axis=1)
        X_test = X_test.drop('demand', axis=1)
        return X_train, X_test, y_train, y_test


    def train_with_cv(self, n_splits_cv=5, early_stopping_rounds=100):
        """
        Trains the LightGBM model using TimeSeriesSplit cross-validation on the pre-computed data.
        """
        
        # Assumes data is coming from feature store and already in correct format
        target = 'demand'
        
        X = self.X_train[self.features]
        y = self.y_train

        tscv = TimeSeriesSplit(n_splits=n_splits_cv)
        fold_rmses = []
        self.ensemble_models = []
        print(f"starting CV with {n_splits_cv} folds...")

        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            print(f"\nFold {fold + 1}:")
            
            X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
            y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
            
            print(f"Training on {len(X_train)} rows.")
            print(f"Validating on {len(X_val)} rows.")

            model = lgb.LGBMRegressor(**self.params)
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(stopping_rounds=early_stopping_rounds)
                ],
                # Use feature names for better logging
                feature_name=self.features,
                categorical_feature=self.categorical_features
            )
            
            y_pred = model.predict(X_val)
            y_pred = np.maximum(0, y_pred).round(0) # Ensure non-negative predictions and nearest whole number

            rmse = np.sqrt(mean_squared_error(y_val, y_pred))
            fold_rmses.append(rmse)
            print(f"  Fold {fold + 1} RMSE: {rmse:.4f}")
            
            self.ensemble_models.append(model)
        
        if not fold_rmses:
            print(f"No valid folds for {self.model_type} evaluation. Model training might be problematic.")
            self.ensemble_models = []
            return [], {}

        avg_rmse = np.mean(fold_rmses)
        print(f"\nAverage CV RMSE for {self.model_type}: {avg_rmse:.4f}")
        self.cv_results = {'avg_rmse': avg_rmse, 'fold_rmses': fold_rmses}

        print(f"\n{len(self.ensemble_models)} {self.model_type} models trained and stored for ensembling.")
        
        return self.ensemble_models, self.cv_results
    

    def log_model(self) -> None:
        """Log the model using MLflow.
           After training we log the model to MLFlow
           registered name specifies which model we are traiing
        """
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run() as run:
            self.run_id = run.info.run_id

            mlflow.set_tag("model_type", self.model_type)
            mlflow.set_tag("features", ",".join(self.features))
            mlflow.set_tag("split_date", str(self.split_date))
            mlflow.set_tag("catalog", self.catalog_name)
            mlflow.set_tag("schema", self.schema_name)

            # Log training and test set
            train_dataset = mlflow.data.from_pandas(self.X_train, name="train")
            mlflow.log_input(train_dataset)
            test_dataset = mlflow.data.from_pandas(self.X_test, name="test")
            mlflow.log_input(test_dataset)

            mlflow.log_metric("avg_rmse", self.cv_results["avg_rmse"])
            for i, rmse in enumerate(self.cv_results["fold_rmses"]):
                mlflow.log_metric(f"fold_{i+1}_rmse", rmse)

            # Average predictions of all models
            ensemble_model = LGBMEnsembleWrapper(self.ensemble_models, self.features)

            # Infer signature
            signature = infer_signature(
                model_input=self.X_train[self.features],
                model_output=ensemble_model.predict(None, self.X_train[self.features])
            )

            # Separate registry paths for baseline vs elasticity
            registered_name = (
                f"{self.catalog_name}.{self.schema_name}.baseline_demand_model"
                if self.model_type == "forecast"
                else f"{self.catalog_name}.{self.schema_name}.elasticity_model"
            )

            self.model_info = mlflow.pyfunc.log_model(
                python_model=ensemble_model,
                artifact_path="lightgbm-ensemble-model",
                signature=signature,
                input_example=self.X_test[self.features].iloc[:1],
                registered_model_name=registered_name,
            )

        def model_improved(self) -> bool:
            """Evaluate the model performance on the test set.

            Compares the current model with the latest registered model using RMSE
            :return: True if the current model performs better, False otherwise.
            """
            client = MlflowClient()
            latest_model_version = client.get_model_version_by_alias(name=self.model_name, alias="latest-model")
            latest_model_uri = f"models:/{latest_model_version.model_id}"

            result = mlflow.models.evaluate(
                latest_model_uri,
                self.eval_data,
                targets=self.config.target,
                model_type="regression",
                evaluators=["default"],
            )
            metrics_old = result.metrics
            if self.metrics["rmse"] >= metrics_old["rmse"]:
                logger.info("Current model performs better. Returning True.")
                return True
            else:
                logger.info("Current model does not improve over latest. Returning False.")
                return False


        def register_model(self) -> None:
            """Register model in Unity Catalog."""
            logger.info("Registering the model in UC...")
            registered_model = mlflow.register_model(
                model_uri=f"runs:/{self.run_id}/lightgbm-pipeline-model",
                name=self.model_name,
                tags=self.tags,
            )
            logger.info(f"Model registered as version {registered_model.version}.")

            latest_version = registered_model.version

            client = MlflowClient()
            client.set_registered_model_alias(
                name=self.model_name,
                alias="latest-model",
                version=latest_version,
            )
            return latest_version



class LGBMEnsembleWrapper(mlflow.pyfunc.PythonModel):
    """
        Wrapper for the CV enemsble of models
        This allows us to log the results
        as a single ensemble of models instead of 
        logging each model separately.
    """
    def __init__(self, models, features):
        self.models = models
        self.features = features

    def predict(self, context, model_input):
        # Select only the expected features
        X = model_input[self.features]

        # Predict with each model
        preds = np.column_stack([m.predict(X) for m in self.models])

        # Average predictions
        final_preds = np.mean(preds, axis=1)

        # Ensure non-negative and integer demand
        return np.maximum(0, final_preds).round(0)
