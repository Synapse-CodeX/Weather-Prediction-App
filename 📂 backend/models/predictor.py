import joblib
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging

from backend.utils.features import create_features_for_prediction
from backend.utils.constraints import apply_physical_constraints

logger = logging.getLogger(__name__)

class WeatherPredictor:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.models = None
        self.load_latest_models()
    
    def load_latest_models(self):
        try:
            self.xgb_model = joblib.load(self.model_dir / "xgboost_model.pkl")
            self.rf_model = joblib.load(self.model_dir / "random_forest_model.pkl")
            self.scaler = joblib.load(self.model_dir / "scaler.pkl")
            self.feature_cols = joblib.load(self.model_dir / "feature_cols.pkl")
            self.targets = joblib.load(self.model_dir / "targets.pkl")
            
            logger.info("✅ Models loaded successfully")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to load models: {e}")
            return False
    
    def predict_single(self, dt: datetime, model_type: str = "ensemble") -> np.ndarray:
        
        if self.xgb_model is None:
            raise RuntimeError("Models not loaded")
        
        features = create_features_for_prediction(dt)
        
        X_pred = pd.DataFrame([features])
        X_pred = X_pred.reindex(columns=self.feature_cols, fill_value=0)
        
        X_scaled = self.scaler.transform(X_pred)
        
        if model_type == "xgb":
            pred = self.xgb_model.predict(X_scaled)[0]
        elif model_type == "rf":
            pred = self.rf_model.predict(X_scaled)[0]
        else:  
            xgb_pred = self.xgb_model.predict(X_scaled)[0]
            rf_pred = self.rf_model.predict(X_scaled)[0]
            pred = 0.2 * xgb_pred + 0.8 * rf_pred
        
        pred = apply_physical_constraints(pred, dt.hour)
        
        return pred
    
    def predict_future(self, hours_ahead: int = 24, model_type: str = "ensemble") -> list:
        
        now = datetime.now()
        future_times = [now + timedelta(hours=i) for i in range(1, hours_ahead + 1)]
        
        predictions = []
        for dt in future_times:
            pred = self.predict_single(dt, model_type)
            
            predictions.append({
                "datetime": dt.strftime("%Y-%m-%d %H:%M:%S"),
                "temperature": float(pred[0]),
                "radiation": float(pred[1]),
                "cloud_coverage": float(pred[2]),
                "rain": float(pred[3]),
                "humidity": float(pred[4]),
                "wind_speed": float(pred[5]),
                "pressure": float(pred[6])
            })
        
        return predictions
    
    def get_model_info(self) -> dict:

        return {
            "model_type": "ensemble",
            "features_count": len(self.feature_cols),
            "targets": self.targets.tolist() if hasattr(self.targets, 'tolist') else self.targets,
            "models_loaded": self.xgb_model is not None
        }