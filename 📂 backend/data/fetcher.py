import requests
import pandas as pd
from datetime import datetime, timedelta
import logging
from typing import Optional

logger = logging.getLogger(__name__)

class WeatherDataFetcher:
    def __init__(self, latitude: float = 21.63, longitude: float = 88.17):
        self.latitude = latitude
        self.longitude = longitude
        self.base_url = "https://archive-api.open-meteo.com/v1/archive"
        
        self.column_mapping = {
            'time': 'Datetime',
            'temperature_2m': 'Temperature(°C)',
            'relativehumidity_2m': 'Relative_Humidity(%)',
            'rain': 'Rain(mm/hour)',
            'cloudcover': 'Cloud_Coverage(%)',
            'shortwave_radiation': 'Radiation(W/m^2)',
            'surface_pressure': 'Pressure(kPa',
            'windspeed_10m': 'Wind_Speed(m/s)'
        }
    
    def fetch_historical_data(self, days: int = 30) -> Optional[pd.DataFrame]:
        
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": start_date.strftime("%Y-%m-%d"),
            "end_date": end_date.strftime("%Y-%m-%d"),
            "hourly": ",".join(self.column_mapping.keys() - {'time'}),
            "timezone": "Asia/Kolkata"
        }
        
        try:
            response = requests.get(self.base_url, params=params, timeout=30)
            response.raise_for_status()
            data = response.json()
            
            if 'hourly' not in data:
                logger.error("No hourly data in response")
                return None
            
            df = pd.DataFrame(data['hourly'])
            df = df.rename(columns=self.column_mapping)
            
            # Convert pressure from hPa to kPa
            df['Pressure(kPa'] = df['Pressure(kPa'] / 10
            
            logger.info(f"Fetched {len(df)} records from {start_date.date()} to {end_date.date()}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data: {e}")
            return None
    
    def fetch_recent_data(self, hours: int = 24) -> Optional[pd.DataFrame]:
        
        return self.fetch_historical_data(days=min(hours//24 + 1, 30))