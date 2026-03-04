from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from datetime import datetime
import logging

from backend.models.predictor import WeatherPredictor
from backend.models.trainer import ModelTrainer
from backend.scheduler.trainer_scheduler import ModelTrainingScheduler
from backend.models.schemas import (
    PredictionRequest, BatchPredictionResponse,
    TrainingStatus, ModelInfo, PredictionResponse
)


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


predictor = WeatherPredictor()
trainer = ModelTrainer()
scheduler = ModelTrainingScheduler()

@asynccontextmanager
async def lifespan(app: FastAPI):
    
    logger.info("🚀 Starting Weather Prediction API...")
    scheduler.start()
    yield
    
    logger.info("👋 Shutting down...")
    scheduler.stop()


app = FastAPI(
    title="Bakkhali Weather Prediction API",
    description="ML-powered weather forecasting for Bakkhali Beach",
    version="1.0.0",
    lifespan=lifespan
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def root():
    return {
        "message": "🌦 Bakkhali Weather Prediction API",
        "version": "1.0.0",
        "status": "operational",
        "docs": "/docs",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "batch-predict": "/batch-predict/{hours}",
            "model-info": "/model-info",
            "train": "/train",
            "training-status": "/training-status"
        }
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": predictor.xgb_model is not None
    }

@app.get("/model-info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the current model"""
    if predictor.xgb_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    info = predictor.get_model_info()
    scheduler_status = scheduler.get_status()
    
    return {
        "active_model_version": scheduler_status["current_model_version"] or "unknown",
        "last_trained": scheduler_status["last_training"],
        "model_type": info["model_type"],
        "features_count": info["features_count"],
        "targets": info["targets"]
    }

@app.get("/predict", response_model=BatchPredictionResponse)
async def predict_weather(
    hours_ahead: int = 24,
    model_type: str = "ensemble"
):
    
    if predictor.xgb_model is None:
        raise HTTPException(status_code=503, detail="Models not loaded")
    
    if hours_ahead < 1 or hours_ahead > 168:
        raise HTTPException(status_code=400, detail="hours_ahead must be between 1 and 168")
    
    if model_type not in ["ensemble", "xgb", "rf"]:
        raise HTTPException(status_code=400, detail="model_type must be ensemble, xgb, or rf")
    
    try:
        predictions = predictor.predict_future(hours_ahead, model_type)
        
        return {
            "predictions": [PredictionResponse(**p) for p in predictions],
            "model_version": scheduler.get_status()["current_model_version"] or "unknown",
            "model_type": model_type,
            "generated_at": datetime.now()
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/train", response_model=TrainingStatus)
async def train_model(background_tasks: BackgroundTasks, days: int = 30):
    
    if days < 7 or days > 365:
        raise HTTPException(status_code=400, detail="days must be between 7 and 365")
    
    def training_task():
        try:
            result = trainer.train_new_model(training_days=days)
            
            predictor.load_latest_models()
            logger.info(f"Training complete: {result['version']}")
        except Exception as e:
            logger.error(f"Training failed: {e}")
    
    background_tasks.add_task(training_task)
    
    return {
        "status": "started",
        "message": f"Training started with {days} days of data",
        "model_version": None,
        "trained_at": None,
        "records_used": None
    }

@app.get("/training-status", response_model=TrainingStatus)
async def get_training_status():
    
    status = scheduler.get_status()
    
    return {
        "status": "idle" if status["is_running"] else "scheduled",
        "message": f"Last training: {status['last_training']}",
        "model_version": status["current_model_version"],
        "trained_at": status["last_training"],
        "records_used": None  # Could be enhanced to store this
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)