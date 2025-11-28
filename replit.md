# DeepRetail: Intelligent Inventory & Markdown Optimization Engine

## Overview
DeepRetail is a Streamlit-based web application designed for retail analytics, sales forecasting, and markdown optimization. It uses the Walmart Store Sales Forecasting dataset to provide intelligent inventory recommendations and analyze promotional effectiveness.

## Current State
The application is fully functional with the following features:
- Data upload interface for Walmart dataset (train.csv, stores.csv, features.csv)
- Demo mode with generated sample data for testing
- Comprehensive EDA dashboard with interactive visualizations
- ML-powered sales forecasting (Random Forest / Gradient Boosting)
- Markdown effectiveness analysis and ROI optimization
- Inventory optimization recommendations with full pagination
- Export functionality for reports

## Project Architecture

### File Structure
```
/
├── app.py                    # Main Streamlit application (850+ lines)
├── utils/
│   ├── __init__.py          # Package init
│   ├── data_processor.py    # Data loading, merging, feature engineering
│   └── forecasting.py       # ML models, prediction, inventory optimization
├── .streamlit/
│   └── config.toml          # Streamlit configuration
├── pyproject.toml           # Python dependencies
└── replit.md                # Project documentation
```

### Key Components

#### Data Processing (utils/data_processor.py)
- `load_and_merge_data()`: Merges train, features, and stores datasets
- `engineer_features()`: Creates time-based and holiday features
- `calculate_store_metrics()`: Aggregates store-level KPIs
- `calculate_holiday_impact()`: Analyzes holiday sales lift
- `calculate_markdown_effectiveness()`: Measures markdown ROI
- `prepare_ml_features()`: Prepares feature matrix for ML models

#### Forecasting (utils/forecasting.py)
- `SalesForecaster`: Class for training RF/GB models with feature tracking
- `prepare_prediction_input()`: Builds properly aligned feature vectors
- `calculate_inventory_recommendations()`: Computes all store-dept combinations
- `calculate_optimal_markdown()`: Markdown strategy optimization

#### Main App (app.py)
- Multi-page navigation with sidebar
- Pages: Home/Upload, EDA Dashboard, Forecasting, Markdown Optimization, Inventory Optimization, Reports
- Session state management for data, model, and pagination persistence
- Interactive visualizations using Plotly

### Key Features

1. **Data Upload**: Upload Walmart dataset files or use demo mode
2. **EDA Dashboard**: 
   - Sales trends and distributions
   - Store performance comparisons
   - Seasonal pattern analysis
   - External factor correlations (temperature, fuel, CPI, unemployment)
3. **Sales Forecasting**:
   - Train Random Forest or Gradient Boosting models
   - View feature importance
   - Make predictions for specific store/dept combinations
4. **Markdown Optimization**:
   - Analyze markdown effectiveness by type
   - Calculate ROI for each markdown category
   - Monthly markdown timing analysis
5. **Inventory Optimization**:
   - Safety stock calculations for ALL store-dept combinations
   - Full pagination with Previous/Next buttons
   - Page size selector (25, 50, 100, 250 per page)
   - Downloadable CSV with all recommendations
   - High volatility alerts
6. **Reports**:
   - Store performance reports
   - Department analysis
   - Markdown reports
   - Full dataset export

## Recent Changes
- **Nov 28, 2025**: Initial implementation of DeepRetail application
  - Created data processing pipeline for Walmart dataset
  - Implemented EDA dashboard with sales trends, seasonal patterns
  - Built ML forecasting with Random Forest and Gradient Boosting
  - Added markdown optimization analysis with ROI calculations
  - Created inventory recommendation engine with full pagination
  - Added sample data generator for demo mode
  - Fixed feature alignment for predictions using prepare_prediction_input
  - Implemented full pagination with Previous/Next buttons and page controls

## Dependencies
- streamlit: Web interface
- pandas, numpy: Data manipulation
- scikit-learn: Machine learning models
- plotly, seaborn, matplotlib: Visualizations

## Running the Application
```bash
streamlit run app.py --server.port 5000
```

## Dataset Requirements
From Kaggle's Walmart Recruiting - Store Sales Forecasting:
- **train.csv**: Historical sales (Store, Dept, Date, Weekly_Sales, IsHoliday)
- **stores.csv**: Store metadata (Store, Type, Size)
- **features.csv**: External factors (Temperature, Fuel_Price, MarkDown1-5, CPI, Unemployment)

Download from: https://www.kaggle.com/competitions/walmart-recruiting-store-sales-forecasting/data

## User Preferences
- None recorded yet
