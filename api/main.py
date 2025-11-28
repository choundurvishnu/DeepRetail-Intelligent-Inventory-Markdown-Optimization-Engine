from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import pandas as pd
import numpy as np
import pickle
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.forecasting import (
    SalesForecaster, 
    PriceElasticityModel, 
    InventoryOptimizer,
    calculate_inventory_recommendations,
    calculate_optimal_markdown
)
from utils.data_processor import load_and_merge_data, engineer_features, prepare_ml_features

app = FastAPI(
    title="DeepRetail API",
    description="Intelligent Inventory & Markdown Optimization Engine API",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_cache: Dict[str, Any] = {}
data_cache: Dict[str, pd.DataFrame] = {}


class PredictionRequest(BaseModel):
    store: int
    dept: int
    month: int
    week: int
    is_holiday: bool = False
    has_markdown: bool = False
    temperature: Optional[float] = 60.0
    fuel_price: Optional[float] = 3.5
    cpi: Optional[float] = 200.0
    unemployment: Optional[float] = 7.0


class PredictionResponse(BaseModel):
    store: int
    dept: int
    predicted_sales: float
    confidence_interval: Dict[str, float]
    model_type: str


class TrainRequest(BaseModel):
    model_type: str = "lightgbm"
    test_size: float = 0.2
    experiment_name: str = "sales_forecasting"


class TrainResponse(BaseModel):
    status: str
    model_type: str
    metrics: Dict[str, float]
    run_id: Optional[str]


class InventoryRequest(BaseModel):
    store: Optional[int] = None
    dept: Optional[int] = None
    service_level: float = 0.95


class InventoryResponse(BaseModel):
    store: int
    dept: int
    avg_weekly_sales: float
    safety_stock: float
    reorder_point: float
    eoq: Optional[float] = None


class ElasticityRequest(BaseModel):
    markdown_type: str
    current_price: float
    proposed_discount_pct: float


class ElasticityResponse(BaseModel):
    markdown_type: str
    elasticity: float
    current_price: float
    proposed_price: float
    expected_sales_change_pct: float
    recommendation: str


class OptimizationRequest(BaseModel):
    initial_price: float
    inventory_level: float
    weeks_remaining: int
    demand_elasticity: float = 1.5


class OptimizationResponse(BaseModel):
    optimal_markdown_pct: float
    new_price: float
    expected_demand_lift_pct: float
    recommendation: str


@app.get("/")
async def root():
    return {
        "message": "DeepRetail API",
        "version": "1.0.0",
        "endpoints": [
            "/health",
            "/train",
            "/predict",
            "/inventory/recommendations",
            "/elasticity/analyze",
            "/optimization/markdown"
        ]
    }


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_loaded": "forecaster" in model_cache,
        "data_loaded": "df" in data_cache
    }


@app.post("/data/load")
async def load_data():
    """Load and process demo data."""
    try:
        from utils.data_processor import generate_sample_data
        
        train_df, stores_df, features_df = generate_sample_data()
        merged_df = load_and_merge_data(train_df, features_df, stores_df)
        df = engineer_features(merged_df)
        
        data_cache['df'] = df
        data_cache['train'] = train_df
        data_cache['stores'] = stores_df
        data_cache['features'] = features_df
        
        return {
            "status": "success",
            "rows": len(df),
            "columns": list(df.columns),
            "stores": df['Store'].nunique(),
            "departments": df['Dept'].nunique()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/train", response_model=TrainResponse)
async def train_model(request: TrainRequest):
    """Train a sales forecasting model."""
    if 'df' not in data_cache:
        raise HTTPException(status_code=400, detail="No data loaded. Call /data/load first.")
    
    try:
        df = data_cache['df']
        X, y, feature_names = prepare_ml_features(df)
        
        if X is None or y is None:
            raise HTTPException(status_code=400, detail="Could not prepare features from data.")
        
        forecaster = SalesForecaster(model_type=request.model_type)
        metrics = forecaster.train(X, y, test_size=request.test_size, 
                                   experiment_name=request.experiment_name)
        
        model_cache['forecaster'] = forecaster
        model_cache['feature_columns'] = forecaster.get_feature_columns()
        
        return TrainResponse(
            status="success",
            model_type=request.model_type,
            metrics=metrics,
            run_id=forecaster.run_id
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a sales prediction for a store-department combination."""
    if 'forecaster' not in model_cache:
        raise HTTPException(status_code=400, detail="No model trained. Call /train first.")
    
    try:
        forecaster = model_cache['forecaster']
        df = data_cache.get('df')
        
        from utils.forecasting import prepare_prediction_input
        
        input_df = prepare_prediction_input(
            df, forecaster, request.store, request.dept,
            request.month, request.week, request.is_holiday, request.has_markdown
        )
        
        prediction = forecaster.predict(input_df)[0]
        
        std_estimate = abs(prediction) * 0.15
        
        return PredictionResponse(
            store=request.store,
            dept=request.dept,
            predicted_sales=round(prediction, 2),
            confidence_interval={
                "lower": round(prediction - 1.96 * std_estimate, 2),
                "upper": round(prediction + 1.96 * std_estimate, 2)
            },
            model_type=forecaster.model_type
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/inventory/recommendations")
async def get_inventory_recommendations(
    store: Optional[int] = None,
    dept: Optional[int] = None,
    limit: int = 100,
    offset: int = 0
):
    """Get inventory optimization recommendations."""
    if 'df' not in data_cache:
        raise HTTPException(status_code=400, detail="No data loaded. Call /data/load first.")
    
    try:
        df = data_cache['df']
        forecaster = model_cache.get('forecaster')
        
        recommendations = calculate_inventory_recommendations(df, forecaster)
        
        if store is not None:
            recommendations = recommendations[recommendations['Store'] == store]
        if dept is not None:
            recommendations = recommendations[recommendations['Dept'] == dept]
        
        total = len(recommendations)
        recommendations = recommendations.iloc[offset:offset + limit]
        
        return {
            "total": total,
            "offset": offset,
            "limit": limit,
            "data": recommendations.to_dict(orient='records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/elasticity/analyze")
async def analyze_elasticity():
    """Analyze price elasticity for markdowns."""
    if 'df' not in data_cache:
        raise HTTPException(status_code=400, detail="No data loaded. Call /data/load first.")
    
    try:
        df = data_cache['df']
        
        elasticity_model = PriceElasticityModel(method='ridge')
        elasticities = elasticity_model.fit(df)
        
        if elasticities is None:
            return {"message": "No markdown data available for elasticity analysis"}
        
        summary = elasticity_model.get_elasticity_summary()
        
        model_cache['elasticity_model'] = elasticity_model
        
        return {
            "elasticities": elasticities,
            "summary": summary.to_dict(orient='records') if summary is not None else []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/optimization/markdown", response_model=OptimizationResponse)
async def optimize_markdown(request: OptimizationRequest):
    """Optimize markdown strategy for clearance."""
    try:
        optimizer = InventoryOptimizer()
        
        result = optimizer.optimize_markdown_timing(
            initial_price=request.initial_price,
            demand_elasticity=request.demand_elasticity,
            inventory_level=request.inventory_level,
            weeks_remaining=request.weeks_remaining
        )
        
        if result['optimal_markdown_pct'] < 10:
            recommendation = "Hold current price - markdown not recommended yet"
        elif result['optimal_markdown_pct'] < 30:
            recommendation = "Moderate markdown recommended to stimulate demand"
        else:
            recommendation = "Aggressive markdown recommended for clearance"
        
        return OptimizationResponse(
            optimal_markdown_pct=round(result['optimal_markdown_pct'], 1),
            new_price=round(result['new_price'], 2),
            expected_demand_lift_pct=round(result['expected_demand_lift'], 1),
            recommendation=recommendation
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/optimization/service-levels")
async def compare_service_levels(
    mean_demand: float,
    std_demand: float
):
    """Compare costs for different service levels."""
    try:
        optimizer = InventoryOptimizer()
        comparison = optimizer.calculate_service_level_cost(
            mean_demand=mean_demand,
            std_demand=std_demand,
            service_levels=[0.90, 0.95, 0.99]
        )
        
        return {
            "comparison": comparison.to_dict(orient='records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/markdown/effectiveness")
async def get_markdown_effectiveness():
    """Get markdown effectiveness analysis."""
    if 'df' not in data_cache:
        raise HTTPException(status_code=400, detail="No data loaded. Call /data/load first.")
    
    try:
        df = data_cache['df']
        analysis = calculate_optimal_markdown(df)
        
        if analysis is None:
            return {"message": "No markdown data available"}
        
        return {
            "analysis": analysis.to_dict(orient='records')
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/experiments")
async def get_experiments():
    """Get MLflow experiment runs."""
    try:
        from utils.forecasting import get_mlflow_experiments
        experiments = get_mlflow_experiments()
        
        return {
            "experiments": experiments.to_dict(orient='records') if len(experiments) > 0 else []
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
