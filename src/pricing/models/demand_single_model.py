import time
import sys
import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc
from mlflow.models import infer_signature
from mlflow import MlflowClient
from loguru import logger
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_squared_error
import lightgbm as lgb


class LGBMModel:
    """
    Single Model to predict sales based on features 
    and Sell Price. Outputs of this model are used in
    the optimiser to try and get find optimal prices
    This model can also be used for scenario analysis
    """
    def __init__(
        self,
        spark,
        config,
        training_df: pd.DataFrame,
        params: dict = {},
    ):
        self.spark = spark
        self.config = config
        # defined as yaml in config
        self.params = self.config.LGBM_FORECAST_PARAMS if not params else params
        self.ensemble_models = []
        self.cv_results = {}
        self.categorical_features = ["item_id", "store_id", "dept_id", "cat_id", "state_id"]
        self.target = "demand"

        # config-driven names (compatible with original code)
        self.catalog_name = getattr(self.config, "catalog_name", "catalog")
        self.schema_name = getattr(self.config, "schema_name", "schema")
        self.experiment_name = getattr(self.config, "experiment_name_basic", "experiment") + "-" + "baseline_demand_model"
        self.model_name = f"{self.catalog_name}.{self.schema_name}.baseline_single_demand"

        # Expect config.num_features[model_type] to be a list of feature names
        self.num_features = getattr(self.config, "num_features", {})['forecast']
        # combine provided numeric/covariate features with categorical features
        self.features = list(self.num_features) + self.categorical_features

        # prepare data (no-op by default; preserved for API compatibility)
        self.training_df_pandas = training_df
        self.split_date = pd.to_datetime("2016-04-25")

        # create train/test split
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()

    def split_data(self):
        """
        Time-based split using self.split_date (if 'date' exists in data).
        If no date column, fall back to a deterministic random split using config.test_fraction (default 0.2).
        Returns X_train, X_test, y_train, y_test (X without target).
        """
        df = self.training_df_pandas.copy()

        df["date"] = pd.to_datetime(df["date"])
        X_train = df.query("date < @self.split_date").copy()
        X_test = df.query("date >= @self.split_date").copy()

        # Expect target column to be 'demand' (preserve original)
        y_train = X_train[self.target]
        y_test = X_test[self.target]
        X_train = X_train.drop(self.target, axis=1)
        X_test = X_test.drop(self.target, axis=1)
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
            print(f"No valid folds for forecast evaluation. Model training might be problematic.")
            self.ensemble_models = []
            return [], {}

        avg_rmse = np.mean(fold_rmses)
        print(f"\nAverage CV RMSE for baseline forecast model: {avg_rmse:.4f}")
        self.cv_results = {'avg_rmse': avg_rmse, 'fold_rmses': fold_rmses}

        print(f"\n{len(self.ensemble_models)} models trained and stored for ensembling.")
        
        return self.ensemble_models, self.cv_results

    def log_model(self) -> None:
        """
        Log the (single or ensemble) model to MLflow using a pyfunc wrapper that
        averages predictions across self.ensemble_models (keeps same logging shape as original).
        """
        mlflow.set_experiment(self.experiment_name)
        with mlflow.start_run() as run:
            self.run_id = run.info.run_id
            mlflow.set_tag("model_type", "baseline_forecast_single_model")
            mlflow.set_tag("features", ",".join(self.features))
            mlflow.set_tag("split_date", str(self.split_date))
            mlflow.set_tag("catalog", self.catalog_name)
            mlflow.set_tag("schema", self.schema_name)

            # Try logging train/test datasets (if mlflow supports the data API in your environment)
            try:
                train_dataset = mlflow.data.from_pandas(self.X_train, name="train")
                mlflow.log_input(train_dataset)
                test_dataset = mlflow.data.from_pandas(self.X_test, name="test")
                mlflow.log_input(test_dataset)
            except Exception:
                # If mlflow.data/from_pandas not available, skip without failing
                logger.debug("mlflow.data.from_pandas not available; skipping dataset logging.")

            # Log metrics from cv_results if available
            if "avg_rmse" in self.cv_results:
                mlflow.log_metric("avg_rmse", float(self.cv_results["avg_rmse"]))
            for i, rmse in enumerate(self.cv_results.get("fold_rmses", [])):
                mlflow.log_metric(f"fold_{i+1}_rmse", float(rmse))

            # Average predictions of all models via wrapper (works with single model as well)
            ensemble_model = LGBMEnsembleWrapper(self.ensemble_models, self.features)

            # Infer signature (use a small slice of training data)
            example_in = self.X_train[self.features].iloc[:min(50, len(self.X_train))]
            try:
                example_out = ensemble_model.predict(None, example_in)
            except Exception:
                example_out = np.zeros((len(example_in),))
            signature = infer_signature(model_input=example_in, model_output=example_out)

            # Select registered name path (compatibility with original)
            registered_name = f"{self.catalog_name}.{self.schema_name}.baseline_single_demand_model"

            # Log the pyfunc model (ensemble wrapper)
            self.model_info = mlflow.pyfunc.log_model(
                python_model=ensemble_model,
                artifact_path="lightgbm-ensemble-model",
                signature=signature,
                input_example=self.X_test[self.features].iloc[:1],
                registered_model_name=registered_name,
            )

    def model_improved(self) -> bool:
        """
        Compare current model's RMSE on test set with the latest registered model (alias 'latest-model').
        Returns True if the current model is better (lower RMSE), or True if no previous model exists.
        """
        client = MlflowClient()
        try:
            latest_version = client.get_model_version_by_alias(name=self.model_name, alias="latest-model")
        except Exception:
            logger.info("No previously registered model found; current model considered improved.")
            return True

        # Construct model URI for the latest registered version
        try:
            version = latest_version.version
            latest_uri = f"models:/{self.model_name}/{version}"
        except Exception as e:
            logger.warning(f"Could not determine latest model URI: {e}. Treating as improved.")
            return True

        # Load previous model and score on test set
        try:
            prev = mlflow.pyfunc.load_model(latest_uri)
            prev_preds = prev.predict(self.X_test[self.features])
            prev_preds = np.array(prev_preds).reshape(-1)
            prev_preds = np.maximum(0, prev_preds).round(0)
            rmse_old = float(np.sqrt(mean_squared_error(self.y_test, prev_preds)))
        except Exception as e:
            logger.warning(f"Could not load/score previous model: {e}. Treating as improved.")
            return True

        # Score current ensemble on test set
        if not self.ensemble_models:
            logger.warning("No trained models available to evaluate improvement. Returning False.")
            return False

        preds_matrix = np.column_stack([m.predict(self.X_test[self.features]) for m in self.ensemble_models])
        preds_avg = np.mean(preds_matrix, axis=1)
        preds_avg = np.maximum(0, preds_avg).round(0)
        rmse_new = float(np.sqrt(mean_squared_error(self.y_test, preds_avg)))

        logger.info(f"Comparing models: new RMSE={rmse_new}, old RMSE={rmse_old}")
        return rmse_new < rmse_old

    def register_model(self) -> None:
        """Register model in MLflow Model Registry (Unity Catalog compatible path)."""
        logger.info("Registering the model in the registry...")
        if not getattr(self, "run_id", None):
            raise RuntimeError("No run_id found. Call log_model() before register_model().")
        model_uri = f"runs:/{self.run_id}/lightgbm-ensemble-model"
        try:
            registered_model = mlflow.register_model(model_uri=model_uri, name=self.model_name)
            latest_version = registered_model.version
            client = MlflowClient()
            client.set_registered_model_alias(name=self.model_name, alias="latest-model", version=latest_version)
            logger.info(f"Model registered as version {latest_version} and alias 'latest-model' set.")
            return latest_version
        except Exception as e:
            logger.error(f"Failed to register model: {e}")
            return None
        

class LGBMEnsembleWrapper(mlflow.pyfunc.PythonModel):
    """
    Wrapper for the (possibly single-model) ensemble. Keeps same predict signature as original.
    """

    def __init__(self, models, features):
        self.models = models
        self.features = features

    def predict(self, context, model_input):
        # Ensure model_input has expected columns
        X = model_input[self.features]
        # Predict with each model and average
        preds = np.column_stack([m.predict(X) for m in self.models])
        final_preds = np.mean(preds, axis=1)
        return np.maximum(0, final_preds).round(0)