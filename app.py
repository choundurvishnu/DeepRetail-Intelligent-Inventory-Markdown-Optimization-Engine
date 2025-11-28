import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(
    page_title="DeepRetail - Inventory & Markdown Optimization",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 1rem;
        color: white;
        text-align: center;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        padding: 10px 20px;
        background-color: #f0f2f6;
        border-radius: 8px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #667eea;
        color: white;
    }
    div[data-testid="stMetricValue"] {
        font-size: 1.8rem;
    }
    .insight-box {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)

from utils.data_processor import (
    load_and_merge_data, engineer_features, calculate_store_metrics,
    calculate_dept_metrics, calculate_holiday_impact, calculate_markdown_effectiveness,
    get_seasonal_patterns, prepare_ml_features
)
from utils.forecasting import (
    SalesForecaster, calculate_inventory_recommendations, calculate_optimal_markdown,
    prepare_prediction_input
)

def init_session_state():
    if 'data_loaded' not in st.session_state:
        st.session_state.data_loaded = False
    if 'merged_data' not in st.session_state:
        st.session_state.merged_data = None
    if 'model_trained' not in st.session_state:
        st.session_state.model_trained = False
    if 'forecaster' not in st.session_state:
        st.session_state.forecaster = None

def render_header():
    st.markdown('<h1 class="main-header">DeepRetail</h1>', unsafe_allow_html=True)
    st.markdown("""
    <p style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 2rem;">
        Intelligent Inventory & Markdown Optimization Engine
    </p>
    """, unsafe_allow_html=True)

def render_data_upload():
    st.header("📁 Data Upload")
    st.markdown("""
    Upload your Walmart dataset files to get started. You'll need:
    - **train.csv** - Historical sales data
    - **stores.csv** - Store information
    - **features.csv** - Additional features (temperature, fuel price, markdowns, etc.)
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        train_file = st.file_uploader("Upload train.csv", type=['csv'], key='train')
    with col2:
        stores_file = st.file_uploader("Upload stores.csv", type=['csv'], key='stores')
    with col3:
        features_file = st.file_uploader("Upload features.csv", type=['csv'], key='features')
    
    if train_file and stores_file and features_file:
        with st.spinner("Processing data..."):
            train_df = pd.read_csv(train_file)
            stores_df = pd.read_csv(stores_file)
            features_df = pd.read_csv(features_file)
            
            merged_data = load_and_merge_data(train_df, features_df, stores_df)
            merged_data = engineer_features(merged_data)
            
            st.session_state.merged_data = merged_data
            st.session_state.data_loaded = True
            
            st.success("Data loaded and processed successfully!")
            
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Records", f"{len(merged_data):,}")
            with col2:
                st.metric("Stores", merged_data['Store'].nunique())
            with col3:
                st.metric("Departments", merged_data['Dept'].nunique())
            with col4:
                st.metric("Date Range", f"{merged_data['Date'].min().strftime('%Y-%m-%d')} to {merged_data['Date'].max().strftime('%Y-%m-%d')}")
    
    if st.button("Load Sample Data (Demo Mode)", type="secondary"):
        with st.spinner("Generating sample data..."):
            sample_data = generate_sample_data()
            st.session_state.merged_data = sample_data
            st.session_state.data_loaded = True
            st.success("Sample data loaded! You can now explore the features.")
            st.rerun()

def generate_sample_data():
    """Generate sample data for demo purposes."""
    np.random.seed(42)
    dates = pd.date_range(start='2010-02-05', end='2012-10-26', freq='W')
    stores = range(1, 46)
    depts = range(1, 100)
    
    records = []
    for store in stores:
        store_base = np.random.uniform(10000, 50000)
        store_type = np.random.choice(['A', 'B', 'C'])
        store_size = {'A': np.random.randint(150000, 220000),
                      'B': np.random.randint(80000, 150000),
                      'C': np.random.randint(30000, 80000)}[store_type]
        
        for dept in np.random.choice(list(depts), size=min(10, len(depts)), replace=False):
            dept_factor = np.random.uniform(0.5, 2.0)
            
            selected_dates = np.random.choice(len(dates), size=min(50, len(dates)), replace=False)
            for date_idx in selected_dates:
                date = dates[date_idx]
                month = date.month
                is_holiday = month in [2, 9, 11, 12] and np.random.random() < 0.3
                
                seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * month / 12)
                holiday_factor = 1.5 if is_holiday else 1.0
                
                base_sales = store_base * dept_factor * seasonal_factor * holiday_factor
                weekly_sales = max(0, base_sales + np.random.normal(0, base_sales * 0.2))
                
                markdown1 = np.random.exponential(5000) if np.random.random() < 0.3 else 0
                markdown2 = np.random.exponential(3000) if np.random.random() < 0.25 else 0
                markdown3 = np.random.exponential(2000) if np.random.random() < 0.2 else 0
                markdown4 = np.random.exponential(1000) if np.random.random() < 0.15 else 0
                markdown5 = np.random.exponential(500) if np.random.random() < 0.1 else 0
                
                records.append({
                    'Store': store,
                    'Dept': dept,
                    'Date': date,
                    'Weekly_Sales': weekly_sales,
                    'IsHoliday': is_holiday,
                    'Temperature': np.random.uniform(30, 100),
                    'Fuel_Price': np.random.uniform(2.5, 4.5),
                    'MarkDown1': markdown1,
                    'MarkDown2': markdown2,
                    'MarkDown3': markdown3,
                    'MarkDown4': markdown4,
                    'MarkDown5': markdown5,
                    'CPI': np.random.uniform(180, 230),
                    'Unemployment': np.random.uniform(5, 14),
                    'Type': store_type,
                    'Size': store_size
                })
    
    df = pd.DataFrame(records)
    df = engineer_features(df)
    return df

def render_eda_dashboard():
    st.header("📊 Exploratory Data Analysis")
    
    if not st.session_state.data_loaded:
        st.warning("Please upload data first to view the analysis.")
        return
    
    df = st.session_state.merged_data
    
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Sales Overview", "🏪 Store Analysis", "📅 Seasonal Patterns", "🌡️ External Factors"])
    
    with tab1:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Sales", f"${df['Weekly_Sales'].sum():,.0f}")
        with col2:
            st.metric("Avg Weekly Sales", f"${df['Weekly_Sales'].mean():,.0f}")
        with col3:
            holiday_impact = calculate_holiday_impact(df)
            st.metric("Holiday Impact", f"+{holiday_impact['impact_percent']:.1f}%")
        with col4:
            st.metric("Peak Sales", f"${df['Weekly_Sales'].max():,.0f}")
        
        st.subheader("Sales Trend Over Time")
        daily_sales = df.groupby('Date')['Weekly_Sales'].sum().reset_index()
        fig = px.line(daily_sales, x='Date', y='Weekly_Sales',
                      title='Total Weekly Sales Trend',
                      labels={'Weekly_Sales': 'Weekly Sales ($)', 'Date': 'Date'})
        fig.update_traces(line_color='#667eea', line_width=2)
        fig.update_layout(hovermode='x unified')
        st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Sales Distribution")
            fig = px.histogram(df, x='Weekly_Sales', nbins=50,
                              title='Distribution of Weekly Sales',
                              labels={'Weekly_Sales': 'Weekly Sales ($)'})
            fig.update_traces(marker_color='#667eea')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Holiday vs Non-Holiday Sales")
            holiday_df = df.groupby('IsHoliday')['Weekly_Sales'].mean().reset_index()
            holiday_df['Type'] = holiday_df['IsHoliday'].map({True: 'Holiday', False: 'Non-Holiday'})
            fig = px.bar(holiday_df, x='Type', y='Weekly_Sales',
                        title='Average Sales: Holiday vs Non-Holiday',
                        labels={'Weekly_Sales': 'Avg Weekly Sales ($)'},
                        color='Type',
                        color_discrete_map={'Holiday': '#764ba2', 'Non-Holiday': '#667eea'})
            st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        store_metrics = calculate_store_metrics(df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Top 10 Stores by Total Sales")
            top_stores = store_metrics.nlargest(10, 'Total_Sales')
            fig = px.bar(top_stores, x='Store', y='Total_Sales',
                        title='Top Performing Stores',
                        labels={'Total_Sales': 'Total Sales ($)', 'Store': 'Store ID'},
                        color='Total_Sales',
                        color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Sales Volatility by Store")
            fig = px.scatter(store_metrics, x='Avg_Sales', y='CV',
                           size='Total_Sales', color='CV',
                           hover_data=['Store'],
                           title='Sales Stability Analysis',
                           labels={'Avg_Sales': 'Average Sales ($)', 'CV': 'Coefficient of Variation (%)'},
                           color_continuous_scale='RdYlGn_r')
            st.plotly_chart(fig, use_container_width=True)
        
        if 'Type' in df.columns:
            st.subheader("Performance by Store Type")
            type_sales = df.groupby('Type')['Weekly_Sales'].agg(['mean', 'sum', 'count']).reset_index()
            type_sales.columns = ['Type', 'Avg_Sales', 'Total_Sales', 'Count']
            
            col1, col2 = st.columns(2)
            with col1:
                fig = px.pie(type_sales, values='Total_Sales', names='Type',
                            title='Sales Distribution by Store Type',
                            color_discrete_sequence=px.colors.sequential.Viridis)
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                fig = px.bar(type_sales, x='Type', y='Avg_Sales',
                            title='Average Sales by Store Type',
                            color='Type',
                            color_discrete_sequence=['#667eea', '#764ba2', '#f093fb'])
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        seasonal = get_seasonal_patterns(df)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Monthly Sales Pattern")
            fig = px.line(seasonal['monthly'], x='Month', y='Avg_Sales',
                         markers=True, title='Average Sales by Month',
                         labels={'Avg_Sales': 'Avg Weekly Sales ($)', 'Month': 'Month'})
            fig.update_traces(line_color='#667eea', line_width=3, marker_size=10)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("Quarterly Sales Pattern")
            fig = px.bar(seasonal['quarterly'], x='Quarter', y='Avg_Sales',
                        title='Average Sales by Quarter',
                        labels={'Avg_Sales': 'Avg Weekly Sales ($)', 'Quarter': 'Quarter'},
                        color='Avg_Sales',
                        color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Weekly Sales Heatmap")
        if 'Year' in df.columns and 'Week' in df.columns:
            pivot = df.pivot_table(values='Weekly_Sales', index='Year', columns='Week', aggfunc='mean')
            fig = px.imshow(pivot, 
                           labels=dict(x="Week of Year", y="Year", color="Avg Sales"),
                           title='Sales Heatmap by Week and Year',
                           color_continuous_scale='Viridis',
                           aspect='auto')
            st.plotly_chart(fig, use_container_width=True)
    
    with tab4:
        st.subheader("Impact of External Factors on Sales")
        
        col1, col2 = st.columns(2)
        with col1:
            if 'Temperature' in df.columns:
                fig = px.scatter(df.sample(min(5000, len(df))), x='Temperature', y='Weekly_Sales',
                               opacity=0.5, title='Temperature vs Sales',
                               labels={'Temperature': 'Temperature (°F)', 'Weekly_Sales': 'Weekly Sales ($)'},
                               trendline='lowess')
                fig.update_traces(marker_color='#667eea')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'Fuel_Price' in df.columns:
                fig = px.scatter(df.sample(min(5000, len(df))), x='Fuel_Price', y='Weekly_Sales',
                               opacity=0.5, title='Fuel Price vs Sales',
                               labels={'Fuel_Price': 'Fuel Price ($)', 'Weekly_Sales': 'Weekly Sales ($)'},
                               trendline='lowess')
                fig.update_traces(marker_color='#764ba2')
                st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if 'CPI' in df.columns:
                fig = px.scatter(df.sample(min(5000, len(df))), x='CPI', y='Weekly_Sales',
                               opacity=0.5, title='Consumer Price Index vs Sales',
                               labels={'CPI': 'CPI', 'Weekly_Sales': 'Weekly Sales ($)'},
                               trendline='lowess')
                fig.update_traces(marker_color='#f093fb')
                st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            if 'Unemployment' in df.columns:
                fig = px.scatter(df.sample(min(5000, len(df))), x='Unemployment', y='Weekly_Sales',
                               opacity=0.5, title='Unemployment Rate vs Sales',
                               labels={'Unemployment': 'Unemployment Rate (%)', 'Weekly_Sales': 'Weekly Sales ($)'},
                               trendline='lowess')
                fig.update_traces(marker_color='#11998e')
                st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Correlation Matrix")
        numeric_cols = ['Weekly_Sales', 'Temperature', 'Fuel_Price', 'CPI', 'Unemployment']
        if 'TotalMarkDown' in df.columns:
            numeric_cols.append('TotalMarkDown')
        available_cols = [col for col in numeric_cols if col in df.columns]
        
        if len(available_cols) > 1:
            corr_matrix = df[available_cols].corr()
            fig = px.imshow(corr_matrix, 
                           labels=dict(color="Correlation"),
                           x=available_cols, y=available_cols,
                           color_continuous_scale='RdBu_r',
                           zmin=-1, zmax=1,
                           title='Feature Correlation Matrix')
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)

def render_forecasting():
    st.header("🔮 Sales Forecasting")
    
    if not st.session_state.data_loaded:
        st.warning("Please upload data first to train the forecasting model.")
        return
    
    df = st.session_state.merged_data
    
    st.markdown("""
    Train a machine learning model to forecast future sales based on historical patterns,
    store characteristics, and external factors.
    """)
    
    col1, col2 = st.columns(2)
    with col1:
        model_type = st.selectbox(
            "Select Model Type",
            options=['random_forest', 'gradient_boosting'],
            format_func=lambda x: 'Random Forest' if x == 'random_forest' else 'Gradient Boosting'
        )
    with col2:
        test_size = st.slider("Test Set Size", 0.1, 0.4, 0.2, 0.05)
    
    if st.button("Train Forecasting Model", type="primary"):
        with st.spinner("Preparing features and training model..."):
            X, y, feature_names = prepare_ml_features(df)
            
            if y is not None and len(y) > 0:
                forecaster = SalesForecaster(model_type=model_type)
                metrics = forecaster.train(X, y, test_size=test_size)
                
                st.session_state.forecaster = forecaster
                st.session_state.model_trained = True
                
                st.success("Model trained successfully!")
                
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("MAE", f"${metrics['mae']:,.0f}")
                with col2:
                    st.metric("RMSE", f"${metrics['rmse']:,.0f}")
                with col3:
                    st.metric("R² Score", f"{metrics['r2']:.3f}")
                with col4:
                    st.metric("MAPE", f"{metrics['mape']:.1f}%")
                
                st.subheader("Feature Importance")
                importance_df = forecaster.get_feature_importance()
                fig = px.bar(importance_df.head(15), x='importance', y='feature',
                            orientation='h', title='Top 15 Most Important Features',
                            labels={'importance': 'Importance', 'feature': 'Feature'},
                            color='importance',
                            color_continuous_scale='Viridis')
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
    
    if st.session_state.model_trained:
        st.divider()
        st.subheader("Make Predictions")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            pred_store = st.selectbox("Select Store", sorted(df['Store'].unique()))
        with col2:
            pred_dept = st.selectbox("Select Department", sorted(df['Dept'].unique()))
        with col3:
            pred_month = st.selectbox("Select Month", range(1, 13))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            is_holiday = st.checkbox("Is Holiday Week?")
        with col2:
            has_markdown = st.checkbox("Has Markdown?")
        with col3:
            pred_week = st.number_input("Week of Year", 1, 52, 1)
        
        if st.button("Predict Sales"):
            try:
                sample_input = prepare_prediction_input(
                    df=df,
                    forecaster=st.session_state.forecaster,
                    store=pred_store,
                    dept=pred_dept,
                    month=pred_month,
                    week=pred_week,
                    is_holiday=is_holiday,
                    has_markdown=has_markdown
                )
                
                prediction = st.session_state.forecaster.predict(sample_input)
                
                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 2rem; border-radius: 1rem; text-align: center; margin-top: 1rem;">
                    <p style="color: white; font-size: 1.2rem; margin: 0;">Predicted Weekly Sales</p>
                    <p style="color: white; font-size: 3rem; font-weight: 700; margin: 0.5rem 0;">${prediction[0]:,.0f}</p>
                    <p style="color: rgba(255,255,255,0.8); font-size: 0.9rem; margin: 0;">
                        Store {pred_store} | Dept {pred_dept} | Month {pred_month}
                    </p>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Error making prediction: {str(e)}")

def render_markdown_optimization():
    st.header("💰 Markdown Optimization")
    
    if not st.session_state.data_loaded:
        st.warning("Please upload data first to analyze markdown effectiveness.")
        return
    
    df = st.session_state.merged_data
    
    if 'TotalMarkDown' not in df.columns:
        st.warning("Markdown data not available in the dataset.")
        return
    
    markdown_effectiveness = calculate_markdown_effectiveness(df)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Avg Sales WITH Markdown", f"${markdown_effectiveness['with_markdown_avg']:,.0f}")
    with col2:
        st.metric("Avg Sales WITHOUT Markdown", f"${markdown_effectiveness['without_markdown_avg']:,.0f}")
    with col3:
        st.metric("Sales Lift", f"+{markdown_effectiveness['sales_lift_percent']:.1f}%")
    
    st.divider()
    
    st.subheader("Markdown Type Analysis")
    optimal_markdown = calculate_optimal_markdown(df)
    
    if optimal_markdown is not None and len(optimal_markdown) > 0:
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(optimal_markdown, x='Markdown_Type', y='ROI',
                        title='ROI by Markdown Type',
                        labels={'ROI': 'Return on Investment', 'Markdown_Type': 'Markdown Type'},
                        color='ROI',
                        color_continuous_scale='RdYlGn')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.bar(optimal_markdown, x='Markdown_Type', y='Sales_Lift',
                        title='Sales Lift by Markdown Type',
                        labels={'Sales_Lift': 'Sales Lift ($)', 'Markdown_Type': 'Markdown Type'},
                        color='Sales_Lift',
                        color_continuous_scale='Viridis')
            st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Markdown Effectiveness Table")
        st.dataframe(optimal_markdown, use_container_width=True)
        
        best_markdown = optimal_markdown.loc[optimal_markdown['ROI'].idxmax()]
        st.markdown(f"""
        <div class="insight-box">
            <h4>💡 Key Insight</h4>
            <p><strong>{best_markdown['Markdown_Type']}</strong> provides the best ROI at <strong>{best_markdown['ROI']:.4f}</strong> 
            with an average sales lift of <strong>${best_markdown['Sales_Lift']:,.0f}</strong>.</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.divider()
    
    st.subheader("Markdown Timing Analysis")
    markdown_by_month = df.groupby('Month').agg({
        'TotalMarkDown': 'sum',
        'Weekly_Sales': 'mean',
        'HasMarkDown': 'mean'
    }).reset_index()
    
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(
        go.Bar(x=markdown_by_month['Month'], y=markdown_by_month['TotalMarkDown'], 
               name='Total Markdown', marker_color='#667eea'),
        secondary_y=False
    )
    fig.add_trace(
        go.Scatter(x=markdown_by_month['Month'], y=markdown_by_month['Weekly_Sales'], 
                   name='Avg Sales', line=dict(color='#764ba2', width=3)),
        secondary_y=True
    )
    fig.update_layout(title='Markdown Spending vs Sales by Month')
    fig.update_xaxes(title_text='Month')
    fig.update_yaxes(title_text='Total Markdown ($)', secondary_y=False)
    fig.update_yaxes(title_text='Avg Weekly Sales ($)', secondary_y=True)
    st.plotly_chart(fig, use_container_width=True)

def render_inventory_optimization():
    st.header("📦 Inventory Optimization")
    
    if not st.session_state.data_loaded:
        st.warning("Please upload data first to get inventory recommendations.")
        return
    
    df = st.session_state.merged_data
    
    if 'inventory_recommendations' not in st.session_state:
        st.session_state.inventory_recommendations = None
    if 'inv_current_page' not in st.session_state:
        st.session_state.inv_current_page = 1
    
    st.markdown("""
    Get intelligent inventory recommendations based on sales patterns, demand volatility,
    and markdown effectiveness.
    """)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        safety_weeks = st.slider("Safety Stock Weeks", 1, 4, 2)
    with col2:
        selected_store = st.selectbox("Focus on Store", ['All'] + list(sorted(df['Store'].unique())))
    with col3:
        page_size = st.selectbox("Records per Page", [25, 50, 100, 250], index=1)
    
    if st.button("Generate Recommendations", type="primary"):
        with st.spinner("Calculating optimal inventory levels..."):
            if selected_store != 'All':
                filtered_df = df[df['Store'] == selected_store]
            else:
                filtered_df = df
            
            recommendations = calculate_inventory_recommendations(
                filtered_df, safety_stock_weeks=safety_weeks
            )
            st.session_state.inventory_recommendations = recommendations
            st.session_state.inv_current_page = 1
    
    recommendations = st.session_state.inventory_recommendations
    
    if recommendations is not None and len(recommendations) > 0:
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Combinations", f"{len(recommendations):,}")
        with col2:
            st.metric("Avg Safety Stock", f"${recommendations['Safety_Stock'].mean():,.0f}")
        with col3:
            st.metric("Avg Reorder Point", f"${recommendations['Reorder_Point'].mean():,.0f}")
        with col4:
            st.metric("Avg Markdown Lift", f"{recommendations['Markdown_Lift'].mean():.1f}%")
        
        st.subheader("Inventory Recommendations by Store-Department")
        
        chart_data = recommendations.head(500) if len(recommendations) > 500 else recommendations
        fig = px.scatter(chart_data, x='Avg_Weekly_Sales', y='Safety_Stock',
                       size='Sales_Volatility', color='Markdown_Lift',
                       hover_data=['Store', 'Dept'],
                       title=f'Safety Stock vs Average Sales (showing {len(chart_data):,} of {len(recommendations):,})',
                       labels={'Avg_Weekly_Sales': 'Avg Weekly Sales ($)',
                               'Safety_Stock': 'Recommended Safety Stock ($)'},
                       color_continuous_scale='RdYlGn')
        st.plotly_chart(fig, use_container_width=True)
        
        st.subheader("Detailed Recommendations")
        
        total_pages = max(1, (len(recommendations) + page_size - 1) // page_size)
        
        st.session_state.inv_current_page = min(st.session_state.inv_current_page, total_pages)
        
        col_prev, col_page_info, col_next = st.columns([1, 3, 1])
        with col_prev:
            if st.button("◀ Previous", disabled=st.session_state.inv_current_page <= 1):
                st.session_state.inv_current_page -= 1
                st.rerun()
        with col_page_info:
            new_page = st.number_input(
                "Go to page",
                min_value=1,
                max_value=total_pages,
                value=st.session_state.inv_current_page,
                key="page_input"
            )
            if new_page != st.session_state.inv_current_page:
                st.session_state.inv_current_page = new_page
                st.rerun()
        with col_next:
            if st.button("Next ▶", disabled=st.session_state.inv_current_page >= total_pages):
                st.session_state.inv_current_page += 1
                st.rerun()
        
        current_page = st.session_state.inv_current_page
        start_idx = (current_page - 1) * page_size
        end_idx = min(start_idx + page_size, len(recommendations))
        display_df = recommendations.iloc[start_idx:end_idx]
        
        st.markdown(f"**Page {current_page} of {total_pages}** | Showing records **{start_idx + 1:,} - {end_idx:,}** of **{len(recommendations):,}** total")
        
        st.dataframe(
            display_df.style.format({
                'Avg_Weekly_Sales': '${:,.0f}',
                'Sales_Volatility': '${:,.0f}',
                'Safety_Stock': '${:,.0f}',
                'Reorder_Point': '${:,.0f}',
                'Markdown_Lift': '{:.1f}%'
            }),
            use_container_width=True,
            height=400
        )
        
        csv = recommendations.to_csv(index=False)
        st.download_button(
            label=f"📥 Download All {len(recommendations):,} Recommendations",
            data=csv,
            file_name="inventory_recommendations.csv",
            mime="text/csv"
        )
        
        high_volatility = recommendations[recommendations['Sales_Volatility'] > recommendations['Sales_Volatility'].quantile(0.75)]
        if len(high_volatility) > 0:
            st.markdown(f"""
            <div class="insight-box">
                <h4>⚠️ High Volatility Alert</h4>
                <p><strong>{len(high_volatility):,}</strong> store-department combinations show high sales volatility. 
                Consider maintaining higher safety stock levels for these departments to prevent stockouts.</p>
            </div>
            """, unsafe_allow_html=True)
    elif recommendations is None:
        st.info("Click 'Generate Recommendations' to calculate inventory optimization suggestions.")

def render_reports():
    st.header("📄 Reports & Export")
    
    if not st.session_state.data_loaded:
        st.warning("Please upload data first to generate reports.")
        return
    
    df = st.session_state.merged_data
    
    st.subheader("Available Reports")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("### Store Performance")
        if st.button("Generate Store Report"):
            store_metrics = calculate_store_metrics(df)
            csv = store_metrics.to_csv(index=False)
            st.download_button(
                label="📥 Download Store Metrics",
                data=csv,
                file_name="store_performance_report.csv",
                mime="text/csv"
            )
    
    with col2:
        st.markdown("### Department Analysis")
        if st.button("Generate Dept Report"):
            dept_metrics = calculate_dept_metrics(df)
            csv = dept_metrics.to_csv(index=False)
            st.download_button(
                label="📥 Download Dept Metrics",
                data=csv,
                file_name="department_analysis_report.csv",
                mime="text/csv"
            )
    
    with col3:
        st.markdown("### Markdown Analysis")
        if st.button("Generate Markdown Report"):
            if 'TotalMarkDown' in df.columns:
                markdown_report = calculate_optimal_markdown(df)
                if markdown_report is not None:
                    csv = markdown_report.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Markdown Report",
                        data=csv,
                        file_name="markdown_analysis_report.csv",
                        mime="text/csv"
                    )
            else:
                st.warning("Markdown data not available.")
    
    st.divider()
    
    st.subheader("Full Dataset Export")
    if st.button("Export Processed Data"):
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Download Full Dataset",
            data=csv,
            file_name="deepretail_processed_data.csv",
            mime="text/csv"
        )
    
    st.divider()
    
    st.subheader("Quick Stats Summary")
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Dataset Overview**")
        st.write(f"- Total Records: {len(df):,}")
        st.write(f"- Date Range: {df['Date'].min().strftime('%Y-%m-%d')} to {df['Date'].max().strftime('%Y-%m-%d')}")
        st.write(f"- Unique Stores: {df['Store'].nunique()}")
        st.write(f"- Unique Departments: {df['Dept'].nunique()}")
    
    with col2:
        st.markdown("**Sales Summary**")
        st.write(f"- Total Sales: ${df['Weekly_Sales'].sum():,.0f}")
        st.write(f"- Average Weekly Sales: ${df['Weekly_Sales'].mean():,.0f}")
        st.write(f"- Max Weekly Sales: ${df['Weekly_Sales'].max():,.0f}")
        st.write(f"- Min Weekly Sales: ${df['Weekly_Sales'].min():,.0f}")

def main():
    init_session_state()
    render_header()
    
    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/shopping-cart.png", width=80)
        st.markdown("## Navigation")
        
        page = st.radio(
            "Select Page",
            options=[
                "🏠 Home & Upload",
                "📊 EDA Dashboard",
                "🔮 Forecasting",
                "💰 Markdown Optimization",
                "📦 Inventory Optimization",
                "📄 Reports"
            ],
            label_visibility="collapsed"
        )
        
        st.divider()
        
        if st.session_state.data_loaded:
            st.success("✅ Data Loaded")
            df = st.session_state.merged_data
            st.caption(f"Records: {len(df):,}")
            st.caption(f"Stores: {df['Store'].nunique()}")
        else:
            st.info("📤 Upload data to begin")
        
        if st.session_state.model_trained:
            st.success("✅ Model Trained")
        
        st.divider()
        st.markdown("### About")
        st.markdown("""
        **DeepRetail** helps retailers optimize 
        inventory levels and markdown 
        strategies using machine learning.
        """)
    
    if page == "🏠 Home & Upload":
        render_data_upload()
    elif page == "📊 EDA Dashboard":
        render_eda_dashboard()
    elif page == "🔮 Forecasting":
        render_forecasting()
    elif page == "💰 Markdown Optimization":
        render_markdown_optimization()
    elif page == "📦 Inventory Optimization":
        render_inventory_optimization()
    elif page == "📄 Reports":
        render_reports()

if __name__ == "__main__":
    main()
