import joblib
import numpy as np
import pandas as pd

xgb_model = joblib.load("model_xgb.pkl")
rf_model = joblib.load("model_rf.pkl")
scaler = joblib.load("scaler.pkl")
feature_cols = joblib.load("feature_cols.pkl")
targets = joblib.load("targets.pkl")

def apply_constraints(pred, hour):
    if hour < 6 or hour > 18:
        pred[1] = 0
    pred[2] = np.clip(pred[2], 0, 100)
    pred[3] = max(0, pred[3])
    pred[4] = np.clip(pred[4], 0, 100)
    pred[5] = max(0, pred[5])
    return pred

def predict_future(df, target_datetime, model_type="ensemble"):
    df = df.sort_values("Datetime").reset_index(drop=True)
    last_row = df.iloc[-1:].copy()
    last_time = df['Datetime'].iloc[-1]

    hours_diff = int((target_datetime - last_time).total_seconds() / 3600)
    if hours_diff <= 0:
        return None

    current_row = last_row.copy()

    for step in range(hours_diff):
        pred_time = last_time + pd.Timedelta(hours=step+1)

        current_row['hour'] = pred_time.hour
        current_row['day'] = pred_time.day
        current_row['month'] = pred_time.month
        current_row['dayofweek'] = pred_time.dayofweek
        current_row['quarter'] = pred_time.quarter
        current_row['dayofyear'] = pred_time.dayofyear

        current_row['hour_sin'] = np.sin(2*np.pi*pred_time.hour/24)
        current_row['hour_cos'] = np.cos(2*np.pi*pred_time.hour/24)

        X_pred = current_row[feature_cols]
        X_scaled = scaler.transform(X_pred)

        if model_type == "xgb":
            pred = xgb_model.predict(X_scaled)[0]
        elif model_type == "rf":
            pred = rf_model.predict(X_scaled)[0]
        else:
            x = xgb_model.predict(X_scaled)[0]
            r = rf_model.predict(X_scaled)[0]
            pred = 0.6*x + 0.4*r

        pred = apply_constraints(pred, pred_time.hour)

        for i, t in enumerate(targets):
            current_row[t] = pred[i]

    return pred
