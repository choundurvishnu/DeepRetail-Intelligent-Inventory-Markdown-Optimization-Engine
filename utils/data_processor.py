import pandas as pd
import numpy as np
from datetime import datetime

def load_and_merge_data(train_df, features_df, stores_df):
    """Load and merge all datasets into a single dataframe."""
    train_df['Date'] = pd.to_datetime(train_df['Date'])
    features_df['Date'] = pd.to_datetime(features_df['Date'])
    
    merged = train_df.merge(features_df, on=['Store', 'Date', 'IsHoliday'], how='left')
    merged = merged.merge(stores_df, on='Store', how='left')
    
    return merged

def engineer_features(df):
    """Create additional features for modeling."""
    df = df.copy()
    
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week.astype(int)
    df['DayOfYear'] = df['Date'].dt.dayofyear
    df['Quarter'] = df['Date'].dt.quarter
    
    df['WeekOfMonth'] = (df['Date'].dt.day - 1) // 7 + 1
    
    df['IsSuperBowl'] = ((df['Month'] == 2) & (df['Week'].isin([6, 7]))).astype(int)
    df['IsLaborDay'] = ((df['Month'] == 9) & (df['Week'].isin([36, 37]))).astype(int)
    df['IsThanksgiving'] = ((df['Month'] == 11) & (df['Week'].isin([47, 48]))).astype(int)
    df['IsChristmas'] = ((df['Month'] == 12) & (df['Week'].isin([52, 53]))).astype(int)
    
    markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    for col in markdown_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)
    
    if all(col in df.columns for col in markdown_cols):
        df['TotalMarkDown'] = df[markdown_cols].sum(axis=1)
        df['HasMarkDown'] = (df['TotalMarkDown'] > 0).astype(int)
    
    if 'Temperature' in df.columns:
        df['Temperature'] = df['Temperature'].fillna(df['Temperature'].mean())
    if 'Fuel_Price' in df.columns:
        df['Fuel_Price'] = df['Fuel_Price'].fillna(df['Fuel_Price'].mean())
    if 'CPI' in df.columns:
        df['CPI'] = df['CPI'].fillna(df['CPI'].mean())
    if 'Unemployment' in df.columns:
        df['Unemployment'] = df['Unemployment'].fillna(df['Unemployment'].mean())
    
    if 'Type' in df.columns:
        df['Type_encoded'] = df['Type'].map({'A': 0, 'B': 1, 'C': 2})
    
    return df

def calculate_store_metrics(df):
    """Calculate store-level performance metrics."""
    store_metrics = df.groupby('Store').agg({
        'Weekly_Sales': ['mean', 'std', 'sum', 'count'],
        'IsHoliday': 'sum'
    }).round(2)
    
    store_metrics.columns = ['Avg_Sales', 'Sales_Std', 'Total_Sales', 'Weeks_Count', 'Holiday_Weeks']
    store_metrics = store_metrics.reset_index()
    
    store_metrics['CV'] = (store_metrics['Sales_Std'] / store_metrics['Avg_Sales'] * 100).round(2)
    
    return store_metrics

def calculate_dept_metrics(df):
    """Calculate department-level performance metrics."""
    dept_metrics = df.groupby(['Store', 'Dept']).agg({
        'Weekly_Sales': ['mean', 'std', 'sum', 'count']
    }).round(2)
    
    dept_metrics.columns = ['Avg_Sales', 'Sales_Std', 'Total_Sales', 'Weeks_Count']
    dept_metrics = dept_metrics.reset_index()
    
    return dept_metrics

def calculate_holiday_impact(df):
    """Calculate the impact of holidays on sales."""
    holiday_sales = df[df['IsHoliday'] == True]['Weekly_Sales'].mean()
    non_holiday_sales = df[df['IsHoliday'] == False]['Weekly_Sales'].mean()
    
    impact = ((holiday_sales - non_holiday_sales) / non_holiday_sales * 100)
    
    return {
        'holiday_avg': holiday_sales,
        'non_holiday_avg': non_holiday_sales,
        'impact_percent': impact
    }

def calculate_markdown_effectiveness(df):
    """Analyze markdown effectiveness."""
    if 'TotalMarkDown' not in df.columns:
        return None
    
    with_markdown = df[df['HasMarkDown'] == 1]['Weekly_Sales'].mean()
    without_markdown = df[df['HasMarkDown'] == 0]['Weekly_Sales'].mean()
    
    if without_markdown > 0:
        lift = ((with_markdown - without_markdown) / without_markdown * 100)
    else:
        lift = 0
    
    markdown_cols = ['MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5']
    markdown_correlation = {}
    for col in markdown_cols:
        if col in df.columns:
            corr = df[[col, 'Weekly_Sales']].corr().iloc[0, 1]
            markdown_correlation[col] = corr
    
    return {
        'with_markdown_avg': with_markdown,
        'without_markdown_avg': without_markdown,
        'sales_lift_percent': lift,
        'markdown_correlations': markdown_correlation
    }

def get_seasonal_patterns(df):
    """Extract seasonal patterns from the data."""
    monthly_sales = df.groupby('Month')['Weekly_Sales'].mean().reset_index()
    monthly_sales.columns = ['Month', 'Avg_Sales']
    
    weekly_sales = df.groupby('Week')['Weekly_Sales'].mean().reset_index()
    weekly_sales.columns = ['Week', 'Avg_Sales']
    
    quarterly_sales = df.groupby('Quarter')['Weekly_Sales'].mean().reset_index()
    quarterly_sales.columns = ['Quarter', 'Avg_Sales']
    
    return {
        'monthly': monthly_sales,
        'weekly': weekly_sales,
        'quarterly': quarterly_sales
    }

def prepare_ml_features(df):
    """Prepare features for machine learning models."""
    feature_cols = [
        'Store', 'Dept', 'Month', 'Week', 'Year', 'Quarter',
        'IsHoliday', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment',
        'Size', 'Type_encoded', 'TotalMarkDown', 'HasMarkDown',
        'IsSuperBowl', 'IsLaborDay', 'IsThanksgiving', 'IsChristmas'
    ]
    
    available_features = [col for col in feature_cols if col in df.columns]
    
    X = df[available_features].copy()
    
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.factorize(X[col])[0]
        X[col] = X[col].fillna(X[col].median())
    
    y = df['Weekly_Sales'].copy() if 'Weekly_Sales' in df.columns else None
    
    return X, y, available_features
