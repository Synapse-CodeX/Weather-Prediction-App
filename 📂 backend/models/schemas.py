from pydantic import BaseModel, Field
from datetime import datetime
from typing import List, Optional

class WeatherDataPoint(BaseModel):
    """Single weather data point from API"""
    datetime: datetime
    temperature: float = Field(..., alias="Temperature(°C)")
    radiation: float = Field(..., alias="Radiation(W/m^2)")
    cloud_coverage: float = Field(..., alias="Cloud_Coverage(%)")
    rain: float = Field(..., alias="Rain(mm/hour)")
    humidity: float = Field(..., alias="Relative_Humidity(%)")
    wind_speed: float = Field(..., alias="Wind_Speed(m/s)")
    pressure: float = Field(..., alias="Pressure(kPa")
    
    class Config:
        populate_by_name = True
        json_schema_extra = {
            "example": {
                "datetime": "2026-02-24 12:00:00",
                "Temperature(°C)": 30.5,
                "Radiation(W/m^2)": 850,
                "Cloud_Coverage(%)": 25,
                "Rain(mm/hour)": 0,
                "Relative_Humidity(%)": 58,
                "Wind_Speed(m/s)": 4.8,
                "Pressure(kPa": 101.0
            }
        }

class PredictionRequest(BaseModel):
    hours_ahead: int = Field(24, ge=1, le=168, description="Hours to predict (1-168)")
    model_type: str = Field("ensemble", pattern="^(ensemble|xgb|rf)$")

class PredictionResponse(BaseModel):
    datetime: datetime
    temperature: float
    radiation: float
    cloud_coverage: float
    rain: float
    humidity: float
    wind_speed: float
    pressure: float

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    model_version: str
    model_type: str
    generated_at: datetime

class TrainingStatus(BaseModel):
    status: str
    message: str
    model_version: Optional[str] = None
    trained_at: Optional[datetime] = None
    records_used: Optional[int] = None

class ModelInfo(BaseModel):
    active_model_version: str
    last_trained: Optional[datetime]
    model_type: str
    features_count: int
    targets: List[str]