import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from scipy import optimize
from scipy.stats import norm
import warnings
import os
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except (ImportError, OSError):
    LIGHTGBM_AVAILABLE = False
    lgb = None

try:
    import mlflow
    import mlflow.sklearn
    if LIGHTGBM_AVAILABLE:
        import mlflow.lightgbm
    MLFLOW_AVAILABLE = True
    mlflow.set_tracking_uri("mlruns")
except ImportError:
    MLFLOW_AVAILABLE = False
    mlflow = None

class SalesForecaster:
    """Sales forecasting model using ensemble methods including LightGBM."""
    
    def __init__(self, model_type='lightgbm'):
        if model_type == 'lightgbm' and not LIGHTGBM_AVAILABLE:
            model_type = 'gradient_boosting'
        self.model_type = model_type
        self.model = None
        self.feature_importance = None
        self.metrics = {}
        self.feature_columns = None
        self.run_id = None
        
    def train(self, X, y, test_size=0.2, experiment_name="sales_forecasting"):
        """Train the forecasting model with optional MLflow tracking."""
        self.feature_columns = list(X.columns)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        if self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
            self.model = lgb.LGBMRegressor(
                n_estimators=200,
                max_depth=12,
                learning_rate=0.05,
                num_leaves=31,
                min_child_samples=20,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=0.1,
                random_state=42,
                n_jobs=-1,
                verbose=-1
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        else:
            self.model = GradientBoostingRegressor(
                n_estimators=100,
                max_depth=8,
                learning_rate=0.1,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42
            )
        
        self.model.fit(X_train, y_train)
        
        y_pred = self.model.predict(X_test)
        
        self.metrics = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'mape': np.mean(np.abs((y_test - y_pred) / (np.abs(y_test) + 1))) * 100
        }
        
        if MLFLOW_AVAILABLE:
            try:
                mlflow.set_experiment(experiment_name)
                with mlflow.start_run() as run:
                    self.run_id = run.info.run_id
                    mlflow.log_param("model_type", self.model_type)
                    mlflow.log_param("test_size", test_size)
                    mlflow.log_param("n_features", len(self.feature_columns))
                    mlflow.log_param("n_samples", len(X))
                    mlflow.log_metrics(self.metrics)
                    
                    if self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
                        mlflow.lightgbm.log_model(self.model, "model")
                    else:
                        mlflow.sklearn.log_model(self.model, "model")
            except Exception:
                pass
        
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
                }).sort_values('importance', ascending=False)
        
        return self.metrics
    
    def predict(self, X):
        """Make predictions using the trained model."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call train() first.")
        return self.model.predict(X)
    
    def get_feature_importance(self):
        """Get feature importance from the trained model."""
        return self.feature_importance
    
    def get_feature_columns(self):
        """Get the list of feature columns used during training."""
        return self.feature_columns
    
    def cross_validate(self, X, y, cv=5):
        """Perform cross-validation."""
        if self.model is None:
            if self.model_type == 'lightgbm' and LIGHTGBM_AVAILABLE:
                self.model = lgb.LGBMRegressor(
                    n_estimators=200, max_depth=12, random_state=42, n_jobs=-1, verbose=-1
                )
            elif self.model_type == 'random_forest':
                self.model = RandomForestRegressor(
                    n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
                )
            else:
                self.model = GradientBoostingRegressor(
                    n_estimators=100, max_depth=8, random_state=42
                )
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='neg_mean_absolute_error')
        return -scores.mean(), scores.std()


class PriceElasticityModel:
    """Price elasticity modeling using regression-based methods."""
    
    def __init__(self, method='ridge'):
        self.method = method
        self.model = None
        self.scaler = StandardScaler()
        self.elasticities = {}
        self.coefficients = None
        
    def fit(self, df, price_cols=['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5'],
            sales_col='Weekly_Sales'):
        """Fit price elasticity model using log-log regression."""
        available_cols = [col for col in price_cols if col in df.columns]
        if not available_cols:
            return None
        
        analysis_df = df[df[sales_col] > 0].copy()
        
        for col in available_cols:
            analysis_df[col] = analysis_df[col].fillna(0)
            analysis_df[col] = analysis_df[col].clip(lower=0)
        
        analysis_df['log_sales'] = np.log1p(analysis_df[sales_col])
        
        for col in available_cols:
            analysis_df[f'log_{col}'] = np.log1p(analysis_df[col])
        
        feature_cols = [f'log_{col}' for col in available_cols]
        
        control_vars = []
        for ctrl in ['Store', 'Dept', 'Month', 'IsHoliday']:
            if ctrl in analysis_df.columns:
                control_vars.append(ctrl)
        
        X = analysis_df[feature_cols + control_vars].copy()
        y = analysis_df['log_sales']
        
        X = X.fillna(0)
        
        X_scaled = self.scaler.fit_transform(X)
        
        if self.method == 'ridge':
            self.model = Ridge(alpha=1.0)
        else:
            self.model = ElasticNet(alpha=0.1, l1_ratio=0.5)
        
        self.model.fit(X_scaled, y)
        
        self.coefficients = pd.DataFrame({
            'feature': X.columns,
            'coefficient': self.model.coef_
        })
        
        for i, col in enumerate(available_cols):
            self.elasticities[col] = self.coefficients.loc[
                self.coefficients['feature'] == f'log_{col}', 'coefficient'
            ].values[0]
        
        return self.elasticities
    
    def get_elasticity_summary(self):
        """Get summary of price elasticities."""
        if not self.elasticities:
            return None
        
        summary = []
        for markdown_type, elasticity in self.elasticities.items():
            interpretation = "elastic" if abs(elasticity) > 1 else "inelastic"
            direction = "increases" if elasticity > 0 else "decreases"
            
            summary.append({
                'Markdown_Type': markdown_type,
                'Elasticity': round(elasticity, 4),
                'Interpretation': interpretation,
                'Effect': f"1% markdown {direction} sales by {abs(elasticity):.2f}%"
            })
        
        return pd.DataFrame(summary)


class InventoryOptimizer:
    """SciPy-based inventory optimization."""
    
    def __init__(self, holding_cost_rate=0.25, stockout_cost_rate=0.40, 
                 order_cost=100, lead_time_weeks=2):
        self.holding_cost_rate = holding_cost_rate
        self.stockout_cost_rate = stockout_cost_rate
        self.order_cost = order_cost
        self.lead_time_weeks = lead_time_weeks
        
    def optimize_reorder_point(self, mean_demand, std_demand, service_level=0.95):
        """Calculate optimal reorder point using statistical optimization."""
        z_score = norm.ppf(service_level)
        
        lead_time_demand = mean_demand * self.lead_time_weeks
        lead_time_std = std_demand * np.sqrt(self.lead_time_weeks)
        
        safety_stock = z_score * lead_time_std
        reorder_point = lead_time_demand + safety_stock
        
        return {
            'reorder_point': reorder_point,
            'safety_stock': safety_stock,
            'lead_time_demand': lead_time_demand,
            'service_level': service_level
        }
    
    def optimize_eoq(self, annual_demand, unit_cost):
        """Calculate Economic Order Quantity."""
        holding_cost = unit_cost * self.holding_cost_rate
        
        if holding_cost <= 0 or annual_demand <= 0:
            return {'eoq': 0, 'total_cost': 0, 'orders_per_year': 0}
        
        eoq = np.sqrt((2 * annual_demand * self.order_cost) / holding_cost)
        
        orders_per_year = annual_demand / eoq if eoq > 0 else 0
        ordering_cost = orders_per_year * self.order_cost
        holding_cost_total = (eoq / 2) * holding_cost
        total_cost = ordering_cost + holding_cost_total
        
        return {
            'eoq': eoq,
            'total_cost': total_cost,
            'orders_per_year': orders_per_year,
            'ordering_cost': ordering_cost,
            'holding_cost': holding_cost_total
        }
    
    def optimize_markdown_timing(self, initial_price, demand_elasticity, 
                                  inventory_level, weeks_remaining, 
                                  min_price_ratio=0.5):
        """Optimize markdown timing and depth using SciPy optimization."""
        
        def objective(x):
            markdown_pct = x[0]
            new_price = initial_price * (1 - markdown_pct)
            
            demand_lift = 1 + (markdown_pct * abs(demand_elasticity))
            
            revenue = new_price * min(inventory_level * demand_lift, inventory_level)
            
            holding_cost = inventory_level * initial_price * self.holding_cost_rate * (weeks_remaining / 52)
            
            return -(revenue - holding_cost)
        
        bounds = [(0, 1 - min_price_ratio)]
        
        result = optimize.minimize(
            objective, 
            x0=[0.1],
            method='L-BFGS-B',
            bounds=bounds
        )
        
        optimal_markdown = result.x[0]
        
        return {
            'optimal_markdown_pct': optimal_markdown * 100,
            'new_price': initial_price * (1 - optimal_markdown),
            'expected_demand_lift': (1 + optimal_markdown * abs(demand_elasticity) - 1) * 100,
            'optimization_success': result.success
        }
    
    def calculate_service_level_cost(self, mean_demand, std_demand, service_levels=[0.90, 0.95, 0.99]):
        """Calculate cost tradeoffs for different service levels."""
        results = []
        
        for sl in service_levels:
            opt = self.optimize_reorder_point(mean_demand, std_demand, sl)
            
            stockout_prob = 1 - sl
            expected_stockout_cost = stockout_prob * mean_demand * self.stockout_cost_rate
            
            holding_cost = opt['safety_stock'] * self.holding_cost_rate
            
            results.append({
                'service_level': sl * 100,
                'safety_stock': opt['safety_stock'],
                'reorder_point': opt['reorder_point'],
                'holding_cost': holding_cost,
                'expected_stockout_cost': expected_stockout_cost,
                'total_cost': holding_cost + expected_stockout_cost
            })
        
        return pd.DataFrame(results)


def calculate_inventory_recommendations(df, forecaster=None, safety_stock_weeks=2):
    """Calculate inventory recommendations based on historical data and optionally forecasts."""
    store_dept_agg = df.groupby(['Store', 'Dept']).agg({
        'Weekly_Sales': ['mean', 'std', 'sum', 'count']
    }).reset_index()
    store_dept_agg.columns = ['Store', 'Dept', 'Avg_Sales', 'Std_Sales', 'Total_Sales', 'Week_Count']
    
    store_dept_agg['Std_Sales'] = store_dept_agg['Std_Sales'].fillna(store_dept_agg['Avg_Sales'] * 0.2)
    
    optimizer = InventoryOptimizer()
    
    optimized_results = []
    for _, row in store_dept_agg.iterrows():
        opt_result = optimizer.optimize_reorder_point(
            row['Avg_Sales'], 
            row['Std_Sales'],
            service_level=0.95
        )
        optimized_results.append({
            'Store': row['Store'],
            'Dept': row['Dept'],
            'Safety_Stock': opt_result['safety_stock'],
            'Reorder_Point': opt_result['reorder_point']
        })
    
    opt_df = pd.DataFrame(optimized_results)
    store_dept_agg = store_dept_agg.merge(opt_df, on=['Store', 'Dept'])
    
    markdown_lift = pd.DataFrame()
    if 'TotalMarkDown' in df.columns and 'HasMarkDown' in df.columns:
        with_md = df[df['HasMarkDown'] == 1].groupby(['Store', 'Dept'])['Weekly_Sales'].mean().reset_index()
        with_md.columns = ['Store', 'Dept', 'Sales_With_MD']
        without_md = df[df['HasMarkDown'] == 0].groupby(['Store', 'Dept'])['Weekly_Sales'].mean().reset_index()
        without_md.columns = ['Store', 'Dept', 'Sales_Without_MD']
        
        markdown_lift = with_md.merge(without_md, on=['Store', 'Dept'], how='outer')
        markdown_lift['Markdown_Lift'] = ((markdown_lift['Sales_With_MD'] / markdown_lift['Sales_Without_MD']) - 1) * 100
        markdown_lift['Markdown_Lift'] = markdown_lift['Markdown_Lift'].fillna(0)
        markdown_lift = markdown_lift[['Store', 'Dept', 'Markdown_Lift']]
    
    if len(markdown_lift) > 0:
        store_dept_agg = store_dept_agg.merge(markdown_lift, on=['Store', 'Dept'], how='left')
        store_dept_agg['Markdown_Lift'] = store_dept_agg['Markdown_Lift'].fillna(0)
    else:
        store_dept_agg['Markdown_Lift'] = 0
    
    recommendations = store_dept_agg[['Store', 'Dept', 'Avg_Sales', 'Std_Sales', 'Safety_Stock', 'Reorder_Point', 'Markdown_Lift']].copy()
    recommendations.columns = ['Store', 'Dept', 'Avg_Weekly_Sales', 'Sales_Volatility', 'Safety_Stock', 'Reorder_Point', 'Markdown_Lift']
    
    recommendations['Store'] = recommendations['Store'].astype(int)
    recommendations['Dept'] = recommendations['Dept'].astype(int)
    recommendations = recommendations.round(2)
    
    recommendations = recommendations.sort_values('Avg_Weekly_Sales', ascending=False).reset_index(drop=True)
    
    return recommendations


def calculate_optimal_markdown(df, target_lift=10):
    """Calculate optimal markdown strategies."""
    if 'TotalMarkDown' not in df.columns:
        return None
    
    markdown_analysis = []
    markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    
    for col in markdown_cols:
        if col not in df.columns:
            continue
            
        col_df = df[df[col] > 0].copy()
        if len(col_df) == 0:
            continue
            
        avg_markdown = col_df[col].mean()
        avg_sales_with = col_df['Weekly_Sales'].mean()
        avg_sales_without = df[df[col] == 0]['Weekly_Sales'].mean()
        
        if avg_markdown > 0:
            roi = (avg_sales_with - avg_sales_without) / avg_markdown
        else:
            roi = 0
        
        markdown_analysis.append({
            'Markdown_Type': col,
            'Avg_Markdown_Amount': round(avg_markdown, 2),
            'Avg_Sales_With': round(avg_sales_with, 2),
            'Avg_Sales_Without': round(avg_sales_without, 2),
            'Sales_Lift': round(avg_sales_with - avg_sales_without, 2),
            'ROI': round(roi, 4)
        })
    
    return pd.DataFrame(markdown_analysis)


def prepare_prediction_input(df, forecaster, store, dept, month, week, is_holiday, has_markdown):
    """Prepare a properly formatted input for prediction."""
    feature_columns = forecaster.get_feature_columns()
    if feature_columns is None:
        raise ValueError("Model feature columns not available")
    
    store_data = df[(df['Store'] == store) & (df['Dept'] == dept)]
    if len(store_data) == 0:
        store_data = df[df['Store'] == store]
    if len(store_data) == 0:
        store_data = df
    
    sample_row = store_data.iloc[0]
    
    input_data = {}
    for col in feature_columns:
        if col == 'Store':
            input_data[col] = store
        elif col == 'Dept':
            input_data[col] = dept
        elif col == 'Month':
            input_data[col] = month
        elif col == 'Week':
            input_data[col] = week
        elif col == 'Year':
            input_data[col] = 2012
        elif col == 'Quarter':
            input_data[col] = (month - 1) // 3 + 1
        elif col == 'IsHoliday':
            input_data[col] = 1 if is_holiday else 0
        elif col == 'HasMarkDown':
            input_data[col] = 1 if has_markdown else 0
        elif col == 'TotalMarkDown':
            input_data[col] = 5000 if has_markdown else 0
        elif col == 'IsSuperBowl':
            input_data[col] = 1 if month == 2 and week in [6, 7] else 0
        elif col == 'IsLaborDay':
            input_data[col] = 1 if month == 9 and week in [36, 37] else 0
        elif col == 'IsThanksgiving':
            input_data[col] = 1 if month == 11 and week in [47, 48] else 0
        elif col == 'IsChristmas':
            input_data[col] = 1 if month == 12 and week in [52, 53] else 0
        elif col in sample_row.index:
            val = sample_row[col]
            if pd.isna(val):
                input_data[col] = df[col].median() if col in df.columns else 0
            else:
                input_data[col] = val
        elif col in df.columns:
            input_data[col] = df[col].median()
        else:
            input_data[col] = 0
    
    input_df = pd.DataFrame([input_data])
    input_df = input_df[feature_columns]
    
    return input_df


def get_mlflow_experiments():
    """Get list of MLflow experiments and runs."""
    if not MLFLOW_AVAILABLE:
        return pd.DataFrame()
    
    try:
        experiments = mlflow.search_experiments()
        runs_data = []
        
        for exp in experiments:
            runs = mlflow.search_runs(experiment_ids=[exp.experiment_id])
            for _, run in runs.iterrows():
                runs_data.append({
                    'experiment': exp.name,
                    'run_id': run['run_id'],
                    'model_type': run.get('params.model_type', 'unknown'),
                    'mae': run.get('metrics.mae', None),
                    'rmse': run.get('metrics.rmse', None),
                    'r2': run.get('metrics.r2', None),
                    'start_time': run.get('start_time', None)
                })
        
        return pd.DataFrame(runs_data)
    except Exception:
        return pd.DataFrame()
