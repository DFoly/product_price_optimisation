import time
import sys
import numpy as np
import pandas as pd
import mlflow
import mlflow.pyfunc
from mlflow.models import infer_signature
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
    def __init__(self, spark, config, params, model_type, training_df: pd.DataFrame):
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
        self.model_name = f"{self.catalog_name}.{self.schema_name}.baseline_demand_model"

        self.num_features =  self.config.num_features[self.model_type]
        self.features = self.num_features + self.categorical_features

        self.training_df_pandas = training_df
        self.split_date = pd.to_datetime("2016-04-25")
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_data()


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



def predict_total_monthly_sales_for_optimization(
    proposed_monthly_price, item_id, store_id, dept_id, cat_id, state_id,
    first_sale_date_item_store, baseline_forecasts_daily_dict, optimization_horizon_dates_list, df_calendar_full,
    elasticity_model_instance, feature_engineer_instance
):
    """
    Predicts total sales for an item-store over the entire optimization horizon (e.g., month),
    given a single proposed price that applies for the whole period.
    This function orchestrates daily predictions using the elasticity model.
    """
    total_predicted_sales_for_month = 0
    
    # Create a temporary DataFrame to hold daily data for the current item-store and horizon
    # This mimics the structure needed for feature engineering, especially lags
    temp_df_for_daily_elasticity = pd.DataFrame(columns=feature_engineer_instance.common_base_features + 
                                                        feature_engineer_instance.categorical_cols_raw +
                                                        ['id', 'date', 'demand', 'first_sale_date', 'event_name_1']) # Include demand for lags

    # Populate temp_df_for_daily_elasticity with context for each day in the horizon
    for current_date_in_horizon in optimization_horizon_dates_list:
        days_since_first_sale_current_day = (current_date_in_horizon - first_sale_date_item_store).days
        
        # If product not released yet, sales are 0. Don't predict.
        if days_since_first_sale_current_day < 0:
            continue

        is_event = df_calendar_full[df_calendar_full['date'] == current_date_in_horizon]['event_name_1'].notna().any().astype(int)
        event_name = df_calendar_full[df_calendar_full['date'] == current_date_in_horizon]['event_name_1'].iloc[0] if is_event else np.nan

        # Get the specific daily baseline forecast for this (item, store, date)
        daily_baseline_forecast = baseline_forecasts_daily_dict.get((item_id, store_id, current_date_in_horizon), 0)

        # Append data for the current day to the temporary DataFrame
        temp_df_for_daily_elasticity = pd.concat([temp_df_for_daily_elasticity, pd.DataFrame([{
            'id': f"{item_id}_{store_id}", # Reconstruct ID for consistency
            'item_id': item_id,
            'store_id': store_id,
            'dept_id': dept_id,
            'cat_id': cat_id,
            'state_id': state_id,
            'date': current_date_in_horizon,
            'first_sale_date': first_sale_date_item_store,
            'sell_price': proposed_monthly_price, # This is the constant price for the month
            'demand': daily_baseline_forecast, # Use baseline as initial demand for lag calculation (will be replaced by elasticity prediction)
            'event_name_1': event_name, # For is_event feature
            'baseline_forecast_feature': daily_baseline_forecast # Crucial feature for elasticity model
        }])], ignore_index=True)

    # Now, process the temp_df_for_daily_elasticity day by day recursively
    # This mimics the recursive prediction logic within the forecasting model, but for elasticity
    
    # Ensure sorted for lag calculations
    temp_df_for_daily_elasticity = temp_df_for_daily_elasticity.sort_values(by=['id', 'date']).reset_index(drop=True)

    max_lookback_days_elasticity = max(feature_engineer_instance.lags) + max(feature_engineer_instance.windows) + 1

    for tdelta in range(len(optimization_horizon_dates_list)): # Loop through the days of the horizon
        day = optimization_horizon_dates_list[tdelta] # Get the current day from the list

        # Select the window of data needed for feature generation for the current 'day'
        # This window includes historical data (actuals or baseline) and previously predicted data.
        tst_window_elasticity = temp_df_for_daily_elasticity[
            (temp_df_for_daily_elasticity['date'] >= day - timedelta(days=max_lookback_days_elasticity)) &
            (temp_df_for_daily_elasticity['date'] <= day)
        ].copy()

        if tst_window_elasticity.empty:
            continue

        # Generate features for the current window for the elasticity model
        tst_window_elasticity_fe = feature_engineer_instance.transform_features(tst_window_elasticity, 'elasticity', is_for_training=False)
        
        # Filter to get only the rows for the current 'day' that need prediction
        tst_to_predict_elasticity = tst_window_elasticity_fe.loc[
            tst_window_elasticity_fe['date'] == day,
            feature_engineer_instance.get_feature_names('elasticity')
        ]

        if tst_to_predict_elasticity.empty:
            continue

        # Predict daily sales using the elasticity model ensemble
        predicted_daily_sales = elasticity_model_instance.predict_ensemble(tst_to_predict_elasticity)
        
        # Update the 'demand' column in the temporary DataFrame with the new prediction
        # This is crucial for subsequent lag calculations within this specific monthly prediction
        temp_df_for_daily_elasticity.loc[temp_df_for_daily_elasticity['date'] == day, 'demand'] = predicted_daily_sales

        total_predicted_sales_for_month += predicted_daily_sales.sum() # Sum up for all items for that day

    return total_predicted_sales_for_month
