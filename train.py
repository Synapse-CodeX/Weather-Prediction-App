import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings("ignore")

from sklearn.preprocessing import RobustScaler
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor


df = pd.read_csv("OUR.csv")   
df['Datetime'] = pd.to_datetime(df['Datetime'])
df = df.sort_values('Datetime').reset_index(drop=True)

print("Initial shape:", df.shape)


df = df.ffill().bfill()

print("Missing values after fill:")
print(df.isnull().sum().sum())



df['hour'] = df['Datetime'].dt.hour
df['day'] = df['Datetime'].dt.day
df['month'] = df['Datetime'].dt.month
df['dayofweek'] = df['Datetime'].dt.dayofweek
df['quarter'] = df['Datetime'].dt.quarter
df['dayofyear'] = df['Datetime'].dt.dayofyear


df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)

df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

df['day_sin'] = np.sin(2 * np.pi * df['dayofyear'] / 365)
df['day_cos'] = np.cos(2 * np.pi * df['dayofyear'] / 365)

df['is_daytime'] = ((df['hour'] >= 6) & (df['hour'] <= 18)).astype(int)



targets = [
    'Temperature(°C)',
    'Radiation(W/m^2)',
    'Cloud_Coverage(%)',
    'Rain(mm/hour)',
    'Relative_Humidity(%)',
    'Wind_Speed(m/s)',
    'Pressure(kPa)'
]


for t in targets:
    if t not in df.columns:
        raise ValueError(f"Missing target column: {t}")


df = df.dropna(subset=targets)

df = df.dropna().reset_index(drop=True)

print("Final dataset shape:", df.shape)



feature_cols = [col for col in df.columns 
                if col not in targets + ['Datetime']]

X = df[feature_cols]
y = df[targets]


scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)


xgb_model = MultiOutputRegressor(
    XGBRegressor(
        n_estimators=200,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
)

rf_model = MultiOutputRegressor(
    RandomForestRegressor(
        n_estimators=200,
        max_depth=12,
        random_state=42,
        n_jobs=-1
    )
)


print("Training XGBoost...")
xgb_model.fit(X_scaled, y)
print("XGBoost trained.")

print("Training Random Forest...")
rf_model.fit(X_scaled, y)
print("Random Forest trained.")


joblib.dump(xgb_model, "model_xgb.pkl")
joblib.dump(rf_model, "model_rf.pkl")
joblib.dump(scaler, "scaler.pkl")
joblib.dump(feature_cols, "feature_cols.pkl")
joblib.dump(targets, "targets.pkl")

print("✅ All models and artifacts saved successfully.")
