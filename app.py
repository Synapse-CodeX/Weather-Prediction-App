import streamlit as st
import pandas as pd
from predictor import predict_future
import plotly.express as px

st.set_page_config(page_title="Bakkhali Weather AI", layout="wide")

st.markdown("""
<style>
.big-title { font-size:40px; font-weight:700; color:#00C9A7; }
.card { padding:20px; border-radius:15px; background-color:#111827; }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="big-title">🌦 Bakkhali Weather Forecast AI</p>', unsafe_allow_html=True)

st.sidebar.header("📂 Upload Data")
uploaded_file = st.sidebar.file_uploader("Upload JSON file", type=["json"])

if uploaded_file:
    df = pd.read_json(uploaded_file)
    df['Datetime'] = pd.to_datetime(df['Datetime'])
    st.success("Data Loaded Successfully")

    col1, col2 = st.columns(2)
    with col1:
        date = st.date_input("Select Date")
    with col2:
        time = st.time_input("Select Time")

    model_choice = st.selectbox("Model", ["ensemble", "xgb", "rf"])

    if st.button("🚀 Predict Future Weather"):
        target_datetime = pd.to_datetime(str(date)+" "+str(time))
        prediction = predict_future(df, target_datetime, model_choice)

        if prediction is None:
            st.error("Target time must be in future.")
        else:
            results = pd.DataFrame({
                "Parameter": [
                    "Temperature (°C)",
                    "Radiation (W/m²)",
                    "Cloud Coverage (%)",
                    "Rain (mm/hr)",
                    "Humidity (%)",
                    "Wind Speed (m/s)",
                    "Pressure (kPa)"
                ],
                "Prediction": prediction
            })

            st.subheader("📊 Prediction Results")
            st.dataframe(results, use_container_width=True)

            fig = px.bar(results, x="Parameter", y="Prediction",
                         color="Prediction",
                         title="Weather Forecast Overview")
            st.plotly_chart(fig, use_container_width=True)
