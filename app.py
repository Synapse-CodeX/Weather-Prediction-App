import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
from datetime import datetime

# Load models and artifacts
@st.cache_resource
def load_models():
    xgb_model = joblib.load("xgboost_model.pkl")
    rf_model = joblib.load("random_forest_model.pkl")
    scaler = joblib.load("scaler.pkl")
    feature_cols = joblib.load("feature_cols.pkl")
    targets = joblib.load("targets.pkl")
    return xgb_model, rf_model, scaler, feature_cols, targets

def apply_constraints(pred, hour):
    pred = list(pred)
    if hour < 6 or hour > 18:
        pred[1] = 0  # Radiation
    pred[2] = np.clip(pred[2], 0, 100)  # Cloud coverage
    pred[3] = max(0, pred[3])  # Rain
    pred[4] = np.clip(pred[4], 0, 100)  # Humidity
    pred[5] = max(0, pred[5])  # Wind speed
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
        
        # Create ALL features that were used in training
        features = {
            'hour': pred_time.hour,
            'day': pred_time.day,
            'month': pred_time.month,
            'dayofweek': pred_time.dayofweek,
            'quarter': pred_time.quarter,
            'dayofyear': pred_time.dayofyear,
            'hour_sin': np.sin(2 * np.pi * pred_time.hour / 24),
            'hour_cos': np.cos(2 * np.pi * pred_time.hour / 24),
            'month_sin': np.sin(2 * np.pi * pred_time.month / 12),
            'month_cos': np.cos(2 * np.pi * pred_time.month / 12),
            'day_sin': np.sin(2 * np.pi * pred_time.dayofyear / 365),
            'day_cos': np.cos(2 * np.pi * pred_time.dayofyear / 365),
            'is_daytime': 1 if 6 <= pred_time.hour <= 18 else 0
        }
        
        # Convert to DataFrame and ensure column order matches feature_cols
        X_pred = pd.DataFrame([features])
        
        # Reindex to match exact feature_cols order (this fills missing columns with 0 if needed)
        X_pred = X_pred.reindex(columns=feature_cols, fill_value=0)
        
        # Scale the features
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

# Streamlit UI
st.set_page_config(page_title="Bakkhali Weather Prediction AI", layout="wide", page_icon="🌦")

# Custom CSS for entire app
st.markdown("""
<style>
/* Global Styles */
.stApp {
    background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
}

/* Header Section */
.header {
    background: rgba(255, 255, 255, 0.1);
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

/* Animations */
@keyframes fadeIn {
    from { opacity: 0; transform: translateY(-20px); }
    to { opacity: 1; transform: translateY(0); }
}

@keyframes float {
    0% { transform: translateY(0px); }
    50% { transform: translateY(-10px); }
    100% { transform: translateY(0px); }
}

/* Sidebar styling */
.css-1d391kg {
    background-color: #1a1a2e;
}

/* Card styling for results */
.stDataFrame {
    border-radius: 10px;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.1);
}

/* Metric cards */
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
# Load models
try:
    xgb_model, rf_model, scaler, feature_cols, targets = load_models()
    models_loaded = True
except:
    st.error("⚠️ Models not found. Please run train.py first to train the models.")
    models_loaded = False

st.sidebar.header("📂 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload JSON file", type=["json"])

if uploaded_file and models_loaded:
    df = pd.read_json(uploaded_file)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    st.success(f"Data Loaded Successfully - {len(df)} records")

    col1, col2 = st.columns(2)
    with col1:
        date = st.date_input("Select Date", min_value=df['Datetime'].max().date())
    with col2:
        time = st.time_input("Select Time")

    model_choice = st.selectbox("Model", ["ensemble (recommended)", "xgb", "rf"])
    
    # Map display name to code name
    model_map = {
        "ensemble (recommended)": "ensemble",
        "xgb": "xgb",
        "rf": "rf"
    }

    if st.button("🚀 Predict Future Weather"):
        target_datetime = pd.to_datetime(str(date)+" "+str(time))
        
        with st.spinner("Calculating predictions..."):
            prediction = predict_future(
                df, target_datetime, model_map[model_choice],
                xgb_model, rf_model, scaler, feature_cols, targets
            )

        if prediction is None:
            st.error("❌ Target time must be in the future.")
        else:
            results = pd.DataFrame({
                "Parameter": [
                    "Temperature (°C)",
                    "Radiation (W/m²)",
                    "Cloud Coverage (%)",
                    "Rain (mm/hr)",
                    "Humidity (%)",
                    "Wind Speed (m/s)",
                    "Pressure (kPa"
                ],
                "Prediction": [round(p, 2) for p in prediction]
            })

            st.subheader("📊 Prediction Results")
            
            # Display metrics in columns
            cols = st.columns(3)
            for i, (param, value) in enumerate(zip(results['Parameter'], results['Prediction'])):
                with cols[i % 3]:
                    st.metric(param, value)
            
            st.dataframe(results, use_container_width=True)

            fig = px.bar(results, x="Parameter", y="Prediction",
                         color="Prediction",
                         title="Weather Forecast Overview",
                         color_continuous_scale="viridis")
            st.plotly_chart(fig, use_container_width=True)
            
            # Show feature importance or additional info
            with st.expander("ℹ️ About this prediction"):
                st.write(f"**Model used:** {model_choice}")
                st.write(f"**Prediction time:** {target_datetime}")
                st.write(f"**Hours ahead:** {(target_datetime - df['Datetime'].max()).total_seconds() / 3600:.1f} hours")
                st.write(f"**Hours ahead:** {(target_datetime - df['Datetime'].max()).total_seconds() / 3600:.1f} hours")

