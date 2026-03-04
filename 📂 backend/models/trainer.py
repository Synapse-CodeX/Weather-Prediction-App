import joblib
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.multioutput import MultiOutputRegressor
from datetime import datetime
import logging
import os
from pathlib import Path

from backend.utils.features import engineer_features
from backend.data.fetcher import WeatherDataFetcher

logger = logging.getLogger(__name__)

class ModelTrainer:
    def __init__(self, model_dir: str = "models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(exist_ok=True)
        self.fetcher = WeatherDataFetcher()
        
        self.xgb_params = {
            'n_estimators': 200,
            'max_depth': 6,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42
        }
        
        self.rf_params = {
            'n_estimators': 200,
            'max_depth': 12,
            'random_state': 42,
            'n_jobs': -1
        }
        
        self.targets = [
            'Temperature(°C)',
            'Radiation(W/m^2)',
            'Cloud_Coverage(%)',
            'Rain(mm/hour)',
            'Relative_Humidity(%)',
            'Wind_Speed(m/s)',
            'Pressure(kPa'
        ]
    
    def train_new_model(self, training_days: int = 30) -> dict:
        
        logger.info(f"Starting model training with {training_days} days of data")
        
        df = self.fetcher.fetch_historical_data(days=training_days)
        if df is None or len(df) < 100:
            raise ValueError("Insufficient data for training")
        
        df = engineer_features(df)
        
        df = df.ffill().bfill()
        
        feature_cols = [col for col in df.columns 
                       if col not in self.targets + ['Datetime']]
        
        X = df[feature_cols]
        y = df[self.targets]
        
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)
        
        logger.info("Training XGBoost model...")
        xgb_model = MultiOutputRegressor(XGBRegressor(**self.xgb_params))
        xgb_model.fit(X_scaled, y)
        
        logger.info("Training Random Forest model...")
        rf_model = MultiOutputRegressor(RandomForestRegressor(**self.rf_params))
        rf_model.fit(X_scaled, y)
        
        version = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        model_files = {
            'xgb': self.model_dir / f"xgb_{version}.pkl",
            'rf': self.model_dir / f"rf_{version}.pkl",
            'scaler': self.model_dir / f"scaler_{version}.pkl",
            'features': self.model_dir / f"features_{version}.pkl",
            'targets': self.model_dir / f"targets_{version}.pkl"
        }
        
        joblib.dump(xgb_model, model_files['xgb'])
        joblib.dump(rf_model, model_files['rf'])
        joblib.dump(scaler, model_files['scaler'])
        joblib.dump(feature_cols, model_files['features'])
        joblib.dump(self.targets, model_files['targets'])
        
        self._update_latest_symlinks(version)
        
        logger.info(f"✅ Models trained and saved (version: {version})")
        
        return {
            'version': version,
            'records_used': len(df),
            'date_range': f"{df['Datetime'].min()} to {df['Datetime'].max()}",
            'feature_cols': feature_cols,
            'targets': self.targets,
            'model_files': model_files
        }
    
    def _update_latest_symlinks(self, version: str):

        import shutil
        
        latest_files = {
            'xgboost_model.pkl': f"xgb_{version}.pkl",
            'random_forest_model.pkl': f"rf_{version}.pkl",
            'scaler.pkl': f"scaler_{version}.pkl",
            'feature_cols.pkl': f"features_{version}.pkl",
            'targets.pkl': f"targets_{version}.pkl"
        }
        
        for latest, versioned in latest_files.items():
            src = self.model_dir / versioned
            dst = self.model_dir / latest
            if src.exists():
                shutil.copy2(src, dst)
                logger.debug(f"Updated {latest}")