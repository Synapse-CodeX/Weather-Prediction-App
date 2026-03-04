import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from datetime import datetime

@st.cache_resource
def load_models():
    xgb_model = joblib.load("models/xgboost_model.pkl")
    rf_model = joblib.load("models/random_forest_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    feature_cols = joblib.load("models/feature_cols.pkl")
    targets = joblib.load("models/targets.pkl")
    return xgb_model, rf_model, scaler, feature_cols, targets

def apply_constraints(pred, hour):
    pred = list(pred)
    if hour < 6 or hour > 18:
        pred[1] = 0
    pred[2] = np.clip(pred[2], 0, 100)
    pred[3] = max(0, pred[3])
    pred[4] = np.clip(pred[4], 0, 100)
    pred[5] = max(0, pred[5])
    return np.array(pred)

def predict_future(df, target_datetime, model_type, xgb_model, rf_model, scaler, feature_cols, targets):
    df = df.sort_values("Datetime").reset_index(drop=True)
    last_time = df['Datetime'].iloc[-1]
    hours_diff = int((target_datetime - last_time).total_seconds() / 3600)
    if hours_diff <= 0:
        return None
    current_prediction = None
    for step in range(hours_diff):
        pred_time = last_time + pd.Timedelta(hours=step+1)
        features = {
            'hour': pred_time.hour, 'day': pred_time.day, 'month': pred_time.month,
            'dayofweek': pred_time.dayofweek, 'quarter': pred_time.quarter,
            'dayofyear': pred_time.dayofyear,
            'hour_sin': np.sin(2 * np.pi * pred_time.hour / 24),
            'hour_cos': np.cos(2 * np.pi * pred_time.hour / 24),
            'month_sin': np.sin(2 * np.pi * pred_time.month / 12),
            'month_cos': np.cos(2 * np.pi * pred_time.month / 12),
            'day_sin': np.sin(2 * np.pi * pred_time.dayofyear / 365),
            'day_cos': np.cos(2 * np.pi * pred_time.dayofyear / 365),
            'is_daytime': 1 if 6 <= pred_time.hour <= 18 else 0
        }
        X_pred = pd.DataFrame([features]).reindex(columns=feature_cols, fill_value=0)
        X_scaled = scaler.transform(X_pred)
        if model_type == "xgb":
            pred = xgb_model.predict(X_scaled)[0]
        elif model_type == "rf":
            pred = rf_model.predict(X_scaled)[0]
        else:
            x = xgb_model.predict(X_scaled)[0]
            r = rf_model.predict(X_scaled)[0]
            pred = 0.2*x + 0.8*r
        pred = apply_constraints(pred, pred_time.hour)
        current_prediction = pred
    return current_prediction

st.set_page_config(page_title="Bakkhali Weather Prediction AI", layout="wide", page_icon="🌦")

st.markdown("""
<style>
.stApp { background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%); }
.header {
    background: rgba(255,255,255,0.1);
    backdrop-filter: blur(10px);
    padding: 30px;
    border-radius: 20px;
    margin-bottom: 30px;
    border: 1px solid rgba(255,255,255,0.2);
    animation: fadeIn 1.5s ease-in;
}
.main-title {
    font-size: 52px;
    font-weight: 800;
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin: 0;
    padding: 0;
}
.emoji-row {
    font-size: 40px;
    text-align: center;
    margin: 10px 0;
    letter-spacing: 15px;
    animation: float 3s ease-in-out infinite;
}
.location-badge {
    text-align: center;
    font-size: 18px;
    color: #e0e0e0;
    margin: 15px auto;
    background: rgba(255,255,255,0.1);
    padding: 8px 25px;
    border-radius: 50px;
    display: inline-block;
    border: 1px solid rgba(255,255,255,0.2);
}
.weather-stats {
    display: flex;
    justify-content: center;
    gap: 40px;
    margin-top: 20px;
    color: #e0e0e0;
    font-size: 16px;
}
.weather-stats span {
    background: rgba(255,255,255,0.05);
    padding: 5px 15px;
    border-radius: 20px;
}
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(-20px); }
    to { opacity: 1; transform: translateY(0); }
}
@keyframes float {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
    100% { transform: translateY(0px); }
}
.css-1d391kg { background-color: #1a1a2e; }
.stDataFrame {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.1);
}
.css-1xarl3l {
    background: rgba(255,255,255,0.05);
    border-radius: 10px;
    padding: 15px;
    border: 1px solid rgba(255,255,255,0.1);
}
</style>

<div class="header">
    <div class="emoji-row">🌡️ ☀️ ☁️ 🌧️ 💧 🌬️</div>
    <h1 class="main-title">🌦 Bakkhali Weather Prediction AI</h1>
    <div style="text-align: center;">
        <span class="location-badge">📍 Bakkhali Beach, West Bengal • 7-Parameter Forecast</span>
    </div>
    <div class="weather-stats">
        <span>🤖 ML-Powered</span>
        <span>⚡ Real-time</span>
        <span>🎯 Multi-Output</span>
        <span>🔮 Ensemble Model</span>
    </div>
</div>
""", unsafe_allow_html=True)

try:
    xgb_model, rf_model, scaler, feature_cols, targets = load_models()
    models_loaded = True
except Exception as e:
    st.error(f"⚠️ Models not found: {e}")
    models_loaded = False

st.sidebar.header("📂 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload JSON file", type=["json"])

if uploaded_file and models_loaded:
    df = pd.read_json(uploaded_file)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    st.success(f"Loaded {len(df)} records")

    col1, col2 = st.columns(2)
    with col1:
        date = st.date_input("Date", min_value=df['Datetime'].max().date())
    with col2:
        time = st.time_input("Time")

    model_choice = st.selectbox("Model", ["ensemble", "xgb", "rf"])
    model_map = {"ensemble": "ensemble", "xgb": "xgb", "rf": "rf"}

    if st.button("🚀 Predict"):
        target = pd.to_datetime(str(date)+" "+str(time))
        with st.spinner("Calculating..."):
            pred = predict_future(df, target, model_map[model_choice],
                                xgb_model, rf_model, scaler, feature_cols, targets)
        if pred is None:
            st.error("❌ Future time required")
        else:
            results = pd.DataFrame({
                "Parameter": ["Temperature (°C)", "Radiation (W/m²)", "Cloud (%)",
                            "Rain (mm/hr)", "Humidity (%)", "Wind (m/s)", "Pressure (kPa"],
                "Prediction": [round(p,2) for p in pred]
            })
            st.subheader("📊 Results")
            cols = st.columns(3)
            for i, (p,v) in enumerate(zip(results['Parameter'], results['Prediction'])):
                with cols[i%3]:
                    st.metric(p, v)
            st.dataframe(results, use_container_width=True)
            fig = px.bar(results, x="Parameter", y="Prediction", color="Prediction",
                        title="Forecast Overview", color_continuous_scale="viridis")
            st.plotly_chart(fig, use_container_width=True)
            with st.expander("ℹ️ Info"):
                st.write(f"Model: {model_choice}")
                st.write(f"Time: {target}")
                st.write(f"Hours ahead: {(target-df['Datetime'].max()).total_seconds()/3600:.1f}")

               
