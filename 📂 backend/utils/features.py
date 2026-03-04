import numpy as np
import pandas as pd
from datetime import datetime

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    
    df = df.copy()
    
    # Time features
    df['hour'] = df['Datetime'].dt.hour
    df['day'] = df['Datetime'].dt.day
    df['month'] = df['Datetime'].dt.month
    df['dayofweek'] = df['Datetime'].dt.dayofweek
    df['quarter'] = df['Datetime'].dt.quarter
    df['dayofyear'] = df['Datetime'].dt.dayofyear
    
    # Cyclical encoding
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
    df['day_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)
    
    # Daytime indicator
    df['is_daytime'] = ((df['hour'] >= 6) & (df['hour'] <= 18)).astype(int)
    
    return df

def create_features_for_prediction(dt: datetime) -> dict:
    
    return {
        'hour': dt.hour,
        'day': dt.day,
        'month': dt.month,
        'dayofweek': dt.weekday(),
        'quarter': (dt.month - 1) // 3 + 1,
        'dayofyear': dt.timetuple().tm_yday,
        'hour_sin': np.sin(2 * np.pi * dt.hour / 24),
        'hour_cos': np.cos(2 * np.pi * dt.hour / 24),
        'month_sin': np.sin(2 * np.pi * dt.month / 12),
        'month_cos': np.cos(2 * np.pi * dt.month / 12),
        'day_sin': np.sin(2 * np.pi * dt.timetuple().tm_yday / 365),
        'day_cos': np.cos(2 * np.pi * dt.timetuple().tm_yday / 365),
        'is_daytime': 1 if 6 <= dt.hour <= 18 else 0
    }