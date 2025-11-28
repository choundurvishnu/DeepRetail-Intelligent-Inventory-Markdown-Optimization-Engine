import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings('ignore')

class SalesForecaster:
    """Sales forecasting model using ensemble methods."""
    
    def __init__(self, model_type='random_forest'):
        self.model_type = model_type
        self.model = None
        self.feature_importance = None
        self.metrics = {}
        self.feature_columns = None
        
    def train(self, X, y, test_size=0.2):
        """Train the forecasting model."""
        self.feature_columns = list(X.columns)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        if self.model_type == 'random_forest':
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
            'mape': np.mean(np.abs((y_test - y_pred) / (y_test + 1))) * 100
        }
        
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        else:
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': [0] * len(X.columns)
            })
        
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
            if self.model_type == 'random_forest':
                self.model = RandomForestRegressor(
                    n_estimators=100, max_depth=15, random_state=42, n_jobs=-1
                )
            else:
                self.model = GradientBoostingRegressor(
                    n_estimators=100, max_depth=8, random_state=42
                )
        
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='neg_mean_absolute_error')
        return -scores.mean(), scores.std()

def calculate_inventory_recommendations(df, forecaster=None, safety_stock_weeks=2):
    """Calculate inventory recommendations based on historical data and optionally forecasts."""
    store_dept_agg = df.groupby(['Store', 'Dept']).agg({
        'Weekly_Sales': ['mean', 'std', 'sum', 'count']
    }).reset_index()
    store_dept_agg.columns = ['Store', 'Dept', 'Avg_Sales', 'Std_Sales', 'Total_Sales', 'Week_Count']
    
    store_dept_agg['Std_Sales'] = store_dept_agg['Std_Sales'].fillna(store_dept_agg['Avg_Sales'] * 0.2)
    
    store_dept_agg['Safety_Stock'] = store_dept_agg['Avg_Sales'] * safety_stock_weeks
    store_dept_agg['Reorder_Point'] = store_dept_agg['Avg_Sales'] * 2 + (1.65 * store_dept_agg['Std_Sales'])
    
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
