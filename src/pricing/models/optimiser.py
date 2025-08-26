
from datetime import timedelta
import pandas as pd
import numpy as np
from ortools.sat.python import cp_model
from typing import Dict, List, Any
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from databricks import feature_store

# Assume these classes are defined elsewhere in the project
from src.pricing.models.baseline_model import LGBMModel # should be getting from MLFlow model registry

class PriceOptimizer:
    """
    """
    def __init__(
        self,
        config: Any,
        fs: Any,
        feature_table_name: str,
        spark: Any,
        df_raw: pd.DataFrame,
        df_calendar_full: pd.DataFrame,
        baseline_forecasts_daily_dict: Dict,
        mlflow_model: Optional[Any] = None,
        model_registry_name: Optional[str] = None,
        price_variation_pct: float = 0.2,
        price_grid_steps: int = 20,
        max_search_seconds: int = 30,
        num_search_workers: int = 4,
        testing: bool = False,
    ):
        self.config = config
        self.fs = fs
        self.feature_table_name = feature_table_name
        self.spark = spark
        # Expect df_raw as pandas DataFrame (preprocessed)
        self.df_raw = df_raw.copy()
        if 'date' in self.df_raw.columns:
            self.df_raw['date'] = pd.to_datetime(self.df_raw['date']).dt.normalize()
        self.df_calendar_full = df_calendar_full.copy()
        if 'date' in self.df_calendar_full.columns:
            self.df_calendar_full['date'] = pd.to_datetime(self.df_calendar_full['date']).dt.normalize()
        self.baseline_forecasts_daily_dict = baseline_forecasts_daily_dict
        self.testing = testing

        # Lags/windows used by feature generation (match your FeatureProducer)
        self.lags = [7, 14, 28]
        self.windows = [7]
        self.max_lookback = max(self.lags) + max(self.windows)  # e.g. 28 + 7 = 35

        # Build optimization context (unique item-store combos)
        self.items_to_optimize = self._get_optimization_context()

        # Categorical features and numeric features (must match training)
        self.categorical_features = ['item_id', 'store_id', 'dept_id', 'cat_id', 'state_id']        
        self.num_feature_names = self.config.num_features['elasticity']

        # Final features given to the elasticity model (must match training)
        self.feature_names = self.num_feature_names + self.categorical_features + ['price_ratio', 'baseline_demand']

        # Load elasticity model (either provided or from MLflow registry)
        if mlflow_model is not None:
            self.elasticity_model = mlflow_model
        else:
            if model_registry_name is None:
                raise ValueError("Provide either mlflow_model or model_registry_name to load the elasticity model")
            model_uri = f"models:/{model_registry_name}/latest"
            self.elasticity_model = mlflow.pyfunc.load_model(model_uri)

        # Price grid / bounds settings
        self.price_variation_pct = price_variation_pct
        self.price_grid_steps = price_grid_steps

        # Solver params
        self.max_search_seconds = max_search_seconds
        self.num_search_workers = num_search_workers

        # Cache for historical lookups for speed (keyed by (item_id, store_id, start_date, end_date))
        self._hist_cache: Dict[Any, pd.DataFrame] = {}

        # Precomputed historical item/store running averages (used as initial values)
        self.historical_item_running_avg = self.df_raw.groupby('item_id')['demand'].mean().to_dict()
        self.historical_store_running_avg = self.df_raw.groupby('store_id')['demand'].mean().to_dict()

    def _get_optimization_context(self) -> List[Dict[str, Any]]:
        """Return a list of dicts with the item-store context for optimization."""
        cols = ['item_id', 'store_id', 'dept_id', 'cat_id', 'state_id', 'first_sale_date']
        available_cols = [c for c in cols if c in self.df_raw.columns]
        unique = self.df_raw[available_cols].drop_duplicates()
        items = []
        for _, row in unique.iterrows():
            items.append({
                'item_id': row['item_id'],
                'store_id': row['store_id'],
                'dept_id': row.get('dept_id'),
                'cat_id': row.get('cat_id'),
                'state_id': row.get('state_id'),
                'first_sale_date': pd.to_datetime(row.get('first_sale_date')) if row.get('first_sale_date') is not None else None
            })
        return items

    def get_baseline_forecast(self, item_id: str, store_id: str, date_obj: pd.Timestamp) -> float:
        """Lookup baseline demand for a specific day."""
        key = (item_id, store_id, pd.to_datetime(date_obj).strftime('%Y-%m-%d'))
        return self.baseline_forecasts_daily_dict.get(key, 0.0)

    def _get_price_bounds(self) -> Dict[str, Dict[str, int]]:
        """
        Determine reasonable integer price ranges for each item based on historical prices.
        Returns mapping item_id -> {'min': int, 'max': int, 'step': int}
        """
        bounds = {}
        for item in self.df_raw['item_id'].unique():
            item_prices = self.df_raw.loc[self.df_raw['item_id'] == item, 'sell_price'].dropna()
            if not item_prices.empty:
                observed_min = float(item_prices.min())
                observed_max = float(item_prices.max())
                # apply variation pct around observed range
                min_p = max(1, int(math.floor(observed_min * (1 - self.price_variation_pct))))
                max_p = int(math.ceil(observed_max * (1 + self.price_variation_pct)))
                # determine reasonable step to limit number of options
                span = max_p - min_p
                if span <= 0:
                    step = 1
                else:
                    step = max(1, int(math.ceil(span / self.price_grid_steps)))
                bounds[item] = {'min': min_p, 'max': max_p, 'step': step}
            else:
                bounds[item] = {'min': 1, 'max': 10, 'step': 1}
        return bounds

    def _get_hist_df(self, item_id: str, store_id: str, start_date: pd.Timestamp, end_date: pd.Timestamp) -> pd.DataFrame:
        """
        Return historical rows for item-store between start_date and end_date inclusive.
        This is cached to avoid repeated work.
        """
        key = (item_id, store_id, pd.to_datetime(start_date).strftime('%Y-%m-%d'),
            pd.to_datetime(end_date).strftime('%Y-%m-%d'))
        if key in self._hist_cache:
            return self._hist_cache[key].copy()

        mask = (
            (self.df_raw['item_id'] == item_id) &
            (self.df_raw['store_id'] == store_id) &
            (self.df_raw['date'] >= pd.to_datetime(start_date)) &
            (self.df_raw['date'] <= pd.to_datetime(end_date))
        )
        hist = self.df_raw.loc[mask].sort_values('date').copy()

        # Ensure continuous date index for computations; fill missing demand with 0 and propagate price forward/backward
        full_idx = pd.date_range(start=start_date, end=end_date, freq='D')
        if 'demand' in hist.columns:
            demand_ser = hist.set_index('date')['demand'].reindex(full_idx)
            # If a long missing period, filling with 0 is a conservative choice; tweak if you prefer different imputation
            demand_ser = demand_ser.fillna(0.0)
        else:
            demand_ser = pd.Series(0.0, index=full_idx)

        if 'sell_price' in hist.columns:
            price_ser = hist.set_index('date')['sell_price'].reindex(full_idx)
            price_ser = price_ser.ffill().bfill().fillna(method='ffill').fillna(1.0)  # fallback to 1.0
        else:
            price_ser = pd.Series(1.0, index=full_idx)

        hist_df = pd.DataFrame({
            'date': full_idx,
            'demand': demand_ser.values,
            'sell_price': price_ser.values
        }, index=full_idx).reset_index(drop=True)

        # attach static categorical columns if available (take last known)
        for cat in ['dept_id', 'cat_id', 'state_id', 'item_id', 'store_id', 'first_sale_date']:
            if cat in hist.columns:
                # pick the last non-null value
                val = hist[cat].dropna().iloc[-1] if not hist[cat].dropna().empty else None
                hist_df[cat] = val
            else:
                # try to pull from df_raw global if present
                any_match = self.df_raw[(self.df_raw['item_id'] == item_id) & (self.df_raw['store_id'] == store_id)]
                if not any_match.empty and cat in any_match.columns:
                    hist_df[cat] = any_match[cat].dropna().iloc[-1]
                else:
                    hist_df[cat] = None

        # cache and return
        self._hist_cache[key] = hist_df.copy()
        return hist_df.copy()


    def _simulate_item_for_price_options(
    self,
    item_ctx: Dict[str, Any],
    price_options: List[float],
    future_dates: List[pd.Timestamp]
) -> Dict[float, float]:
    """
    For a single item-store, simulate across the given candidate price options.
    Returns mapping price -> total_predicted_sales_over_horizon (float).
    The method vectorizes model.predict across price options per day.
    """
    item_id = item_ctx['item_id']
    store_id = item_ctx['store_id']
    first_sale_date = item_ctx.get('first_sale_date', None)
    # historical range needed to compute lags/rolling windows
    lookback_days = self.max_lookback
    hist_start = (future_dates[0] - timedelta(days=lookback_days)).normalize()
    hist_end = (future_dates[0] - timedelta(days=1)).normalize()
    hist_df = self._get_hist_df(item_id, store_id, hist_start, hist_end)
    # create a base demand series indexed by date for historical context
    base_dates = pd.to_datetime(hist_df['date'])
    base_demand = pd.Series(hist_df['demand'].values, index=base_dates)
    base_price = pd.Series(hist_df['sell_price'].values, index=base_dates)
    # initial baseline price sum & count for each price option (expanding mean up to previous day)
    hist_price_sum = float(base_price.sum())
    hist_price_count = int(len(base_price))
    if hist_price_count == 0:
        hist_price_sum = 0.0
        hist_price_count = 1  # avoid div by zero, baseline price 0 -> fallback to proposed price
    # initial store running avg/demand sum & counts (we update these per price option separately)
    hist_store_demand_sum = float(base_demand.sum())
    hist_store_count = int(len(base_demand))
    # initial item running avg (we do not update across stores here; optional improvement later)
    item_running_avg_init = float(self.historical_item_running_avg.get(item_id, base_demand.mean() if not base_demand.empty else 0.0))
    # initial store running avg
    store_running_avg_init = float(self.historical_store_running_avg.get(store_id, base_demand.mean() if not base_demand.empty else 0.0))
    # For each price option, maintain its own demand series (pandas Series) and baseline sums & counts
    n_prices = len(price_options)
    demand_series_list = [base_demand.copy() for _ in range(n_prices)]
    price_sum_list = [hist_price_sum for _ in range(n_prices)]
    price_count_list = [hist_price_count for _ in range(n_prices)]
    store_sum_list = [hist_store_demand_sum for _ in range(n_prices)]
    store_count_list = [hist_store_count for _ in range(n_prices)]
    total_sales = np.zeros(n_prices, dtype=float)

    # Precompute a few static values to avoid recomputation in loops
    item_id_str = str(item_id)
    store_id_str = str(store_id)
    dept_id = item_ctx.get('dept_id')
    cat_id = item_ctx.get('cat_id')
    state_id = item_ctx.get('state_id')

    # Simulate day-by-day; per day do a single call to model.predict with a row per price option
    for day in future_dates:
        day = pd.to_datetime(day).normalize()
        rows = []
        for i, p in enumerate(price_options):
            ds = demand_series_list[i]
            # compute lag features: lag_t7, lag_t14, lag_t28
            feats = {}
            for lag in self.lags:
                lag_date = (day - timedelta(days=lag)).normalize()
                lag_val = ds.get(lag_date, np.nan)
                feats[f'lag_t{lag}'] = float(lag_val) if not (pd.isna(lag_val)) else np.nan
                # rolling mean of lag feature over previous window w: mean of demand from (day - lag - w) .. (day - lag - 1)
                for w in self.windows:
                    start = (day - timedelta(days=lag + w)).normalize()
                    end = (day - timedelta(days=lag + 1)).normalize()
                    if start <= end:
                        slice_vals = ds.loc[start:end] if (start in ds.index or end in ds.index or True) else pd.Series(dtype=float)
                        # use .loc with slice will produce range even if missing dates; reindex would be slower; just take intersection
                        try:
                            slice_vals = ds.loc[start:end]
                        except Exception:
                            slice_vals = ds[(ds.index >= start) & (ds.index <= end)]
                        feats[f'rolling_mean_lag{lag}_w{w}'] = float(slice_vals.mean()) if not slice_vals.empty else np.nan
                    else:
                        feats[f'rolling_mean_lag{lag}_w{w}'] = np.nan
            # price and baseline_price / price_ratio
            baseline_price_current = price_sum_list[i] / price_count_list[i] if price_count_list[i] > 0 else float(p)
            price_ratio = float(p) / baseline_price_current if baseline_price_current > 0 else 1.0
            # baseline demand lookup
            baseline_demand_val = float(self.get_baseline_forecast(item_id_str, store_id_str, day))
            # running averages
            item_running_avg_val = float(item_running_avg_init)
            store_running_avg_val = float(store_sum_list[i] / store_count_list[i]) if store_count_list[i] > 0 else float(store_running_avg_init)
            # days_since_first_sale
            if first_sale_date is not None:
                try:
                    first_sale_dt = pd.to_datetime(first_sale_date)
                    days_since_first_sale = int((day - first_sale_dt).days)
                except Exception:
                    days_since_first_sale = np.nan
            else:
                days_since_first_sale = np.nan
            # date/time features - match training (use isoweekday 1..7)
            feats.update({
                'sell_price': float(p),
                'item_running_avg': item_running_avg_val,
                'store_running_avg': store_running_avg_val,
                'days_since_first_sale': days_since_first_sale,
                'dayofweek': int(day.isoweekday()),
                'month': int(day.month),
                'year': int(day.year),
                'week': int(day.isocalendar()[1]),
                'dayofyear': int(day.dayofyear),
                'is_event': int(self.df_calendar_full.loc[self.df_calendar_full['date'] == day, 'event_name_1'].notna().any()),
                # categorical features
                'item_id': item_id_str,
                'store_id': store_id_str,
                'dept_id': dept_id,
                'cat_id': cat_id,
                'state_id': state_id,
                # elasticity-specific
                'price_ratio': price_ratio,
                'baseline_demand': baseline_demand_val
            })
            # re-order / filter later when creating DataFrame to match feature_names
            rows.append(feats)
        # Build DataFrame for the day's price options and call model once
        X_day = pd.DataFrame(rows)
        # Ensure all expected feature columns exist in the same order as model
        for col in self.feature_names:
            if col not in X_day.columns:
                X_day[col] = np.nan
        X_day = X_day[self.feature_names]
        # predict (mlflow pyfunc or any model that accepts pandas DataFrame)
        preds = self.elasticity_model.predict(X_day)
        # convert to numpy array of floats
        if isinstance(preds, pd.Series):
            preds_arr = preds.values.astype(float)
        elif isinstance(preds, (np.ndarray, list)):
            preds_arr = np.array(preds, dtype=float).reshape(-1)
        else:
            # fallback
            preds_arr = np.array(preds).astype(float).reshape(-1)
        # Ensure non-negative predictions
        preds_arr = np.maximum(0.0, preds_arr)
        # Update per-price demand series and accounting
        for i, p in enumerate(price_options):
            pred = float(preds_arr[i])
            # append predicted demand for this future day to the demand series for this price option
            demand_series_list[i].at[day] = pred
            # update baseline price expanding mean for next day's baseline_price
            price_sum_list[i] += float(p)
            price_count_list[i] += 1
            # update store running average (we treat per-store separately)
            store_sum_list[i] += pred
            store_count_list[i] += 1
            # accumulate total sales for the horizon
            total_sales[i] += pred
    # return mapping price -> total_sales
    return {price_options[i]: float(total_sales[i]) for i in range(n_prices)}


    def _generate_price_demand_curves(self, future_dates: List[pd.Timestamp], item_price_bounds: Dict[str, Any]) -> Dict[tuple, Dict[float, float]]:
        """
        For each item-store in the optimization context, generate a price -> total_sales mapping.
        Returns dict keyed by (item_id, store_id) -> {price -> total_sales}
        """
        curves = {}
        # Optionally limit items in testing mode
        contexts = self.items_to_optimize if not self.testing else self.items_to_optimize[: min(200, len(self.items_to_optimize))]
        for item_ctx in contexts:
            item_id = item_ctx['item_id']
            store_id = item_ctx['store_id']
            bounds = item_price_bounds.get(item_id, {'min': 1, 'max': 10, 'step': 1})
            price_options = list(range(bounds['min'], bounds['max'] + 1, bounds['step']))
            if len(price_options) == 0:
                price_options = [bounds.get('min', 1)]
            try:
                curve = self._simulate_item_for_price_options(item_ctx, price_options, future_dates)
            except Exception as e:
                # On simulation failure, fallback to zero demand mapping to avoid crash
                curve = {p: 0.0 for p in price_options}
                print(f"Warning: simulation failed for {item_id}_{store_id}: {e}")
            curves[(item_id, store_id)] = curve
        return curves


    def optimize(self, future_dates: List[pd.Timestamp], optimization_duration_days: int) -> pd.DataFrame:
        """
        Compute optimal monthly price for each item-store for the given horizon.
        - future_dates: list of pandas Timestamp objects for each day in the optimization horizon
        - optimization_duration_days is not used explicitly (in case you want to assert len(future_dates)==optimization_duration_days)
        Returns pandas DataFrame with columns:
            ['item_id', 'store_id', 'optimal_price', 'predicted_monthly_sales', 'predicted_monthly_revenue']
        """
        # 1) Build price bounds
        item_price_bounds = self._get_price_bounds()
        # 2) Precompute price -> demand curves
        price_demand_curves = self._generate_price_demand_curves(future_dates, item_price_bounds)
        # 3) Build CP-SAT model with binary choice variables where revenue coefficients are precomputed constants
        model = cp_model.CpModel()
        choice_vars = {}  # (item,store,price) -> BoolVar
        revenue_terms = []  # list of tuples (boolVar, revenue_cents)
        # If you want additional constraints (e.g., limit # of discounts), add them here
        for (item_id, store_id), curve in price_demand_curves.items():
            price_options = sorted(curve.keys())
            if not price_options:
                continue
            bool_vars = []
            for p in price_options:
                var_name = f"x_{str(item_id)}_{str(store_id)}_{str(p)}"
                b = model.NewBoolVar(var_name)
                choice_vars[(item_id, store_id, p)] = b
                predicted_sales = float(curve[p])
                # Compute integer revenue coefficient in cents to avoid floats in objective
                revenue_cents = int(round(predicted_sales * float(p) * 100.0))
                revenue_terms.append((b, revenue_cents))
                bool_vars.append(b)
            # exactly one price option must be chosen
            model.Add(sum(bool_vars) == 1)
        # Objective: maximize total revenue (sum revenue_cents * chosen_bool)
        objective_terms = []
        for b, rev in revenue_terms:
            objective_terms.append(rev * b)
        model.Maximize(sum(objective_terms))
        # Solver config
        solver = cp_model.CpSolver()
        solver.parameters.max_time_in_seconds = float(self.max_search_seconds)
        solver.parameters.num_search_workers = int(self.num_search_workers)
        status = solver.Solve(model)
        if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
            print("Solver did not return a feasible/optimal solution. Status:", status)
            return pd.DataFrame(columns=['item_id', 'store_id', 'optimal_price', 'predicted_monthly_sales', 'predicted_monthly_revenue'])
        # 4) Extract results and produce DataFrame
        results = []
        for (item_id, store_id), curve in price_demand_curves.items():
            selected_price = None
            for p in sorted(curve.keys()):
                b = choice_vars.get((item_id, store_id, p))
                if b is None:
                    continue
                if solver.Value(b) == 1:
                    selected_price = p
                    break
            if selected_price is None:
                # fallback: choose highest revenue option
                selected_price = max(curve.items(), key=lambda kv: kv[1] * kv[0])[0] if curve else None
            predicted_sales = float(curve.get(selected_price, 0.0)) if selected_price is not None else 0.0
            predicted_revenue = float(predicted_sales * (selected_price if selected_price is not None else 0.0))
            results.append({
                'item_id': item_id,
                'store_id': store_id,
                'optimal_price': selected_price,
                'predicted_monthly_sales': predicted_sales,
                'predicted_monthly_revenue': predicted_revenue
            })
        return pd.DataFrame(results)