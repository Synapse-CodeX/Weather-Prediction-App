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
        
        features = {
            'hour': pred_time.hour,
            'day': pred_time.day,
            'month': pred_time.month,
            'dayofweek': pred_time.dayofweek,
            'quarter': pred_time.quarter,
            'dayofyear': pred_time.dayofyear,
            'hour_sin': np.sin(2*np.pi*pred_time.hour/24),
            'hour_cos': np.cos(2*np.pi*pred_time.hour/24),
            'month_sin': np.sin(2*np.pi*pred_time.month/12),
            'month_cos': np.cos(2*np.pi*pred_time.month/12),
            'day_sin': np.sin(2*np.pi*pred_time.dayofyear/365),
            'day_cos': np.cos(2*np.pi*pred_time.dayofyear/365),
            'is_daytime': 1 if 6 <= pred_time.hour <= 18 else 0
        }
        
        X_pred = pd.DataFrame([features])[feature_cols]
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
st.set_page_config(page_title="Bakkhali Weather Prediction AI", layout="wide")

st.markdown("""
<style>
.big-title { font-size:40px; font-weight:700; color:#00C9A7; }
.card { padding:20px; border-radius:15px; background-color:#111827; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">🌦 Bakkhali Weather Forecast AI</p>', unsafe_allow_html=True)

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
<<<<<<< HEAD
                st.write(f"**Hours ahead:** {(target_datetime - df['Datetime'].max()).total_seconds() / 3600:.1f} hours")
=======
                st.write(f"**Hours ahead:** {(target_datetime - df['Datetime'].max()).total_seconds() / 3600:.1f} hours")
>>>>>>> 5670237e5d521ae03cd8a6aed2ba6c680e5d36d0
