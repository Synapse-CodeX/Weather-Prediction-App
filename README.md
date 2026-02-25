# 🌦 Weather Forecast App  

An end-to-end **Multi-Output Time Series Weather Forecasting System** built using Machine Learning and deployed as an interactive Streamlit web application.

This system predicts future weather conditions for Bakkhali using advanced feature engineering, ensemble modeling, and physical constraint enforcement.

---

## 🚀 Project Overview

This project performs:

- 📊 Time-series feature engineering  
- 🔁 Lag and rolling statistical features  
- 🌗 Cyclical time encoding (sin/cos transformation)  
- ⚖ Robust scaling  
- 🌲 Random Forest & ⚡ XGBoost multi-output regression  
- 🧠 Ensemble prediction  
- 🌤 Physical constraint correction  
- 🌐 Streamlit web deployment  

The system predicts:

- 🌡 Temperature (°C)  
- ☀ Radiation (W/m²)  
- ☁ Cloud Coverage (%)  
- 🌧 Rain (mm/hour)  
- 💧 Relative Humidity (%)  
- 🌬 Wind Speed (m/s)  
- 🧭 Pressure (kPa)  

---

## 🏗 System Architecture

```
JSON Input 
    ↓
Feature Engineering
    ↓
Robust Scaling
    ↓
Multi-Output Models (XGBoost + Random Forest)
    ↓
Ensemble Combination
    ↓
Physical Constraints
    ↓
Streamlit Visualization
```

---

## 📂 Project Structure

```
Weather-Prediction-App/
│
├── app.py
├── model_xgb.pkl
├── model_rf.pkl
├── scaler.pkl
├── feature_cols.pkl
├── targets.pkl
├── .streamlit
    └── config.toml
├── requirements.txt
└── README.md
└── LICENSE
```

---

## ⚙ Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/Weather-Prediction-App.git
cd Weather-Prediction-App
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## 🧠 Training the Model

Before running the app, generate model files:

```bash
python train.py
```

This will create:

- model_xgb.pkl  
- model_rf.pkl  
- scaler.pkl  
- feature_cols.pkl  
- targets.pkl  

These files are required for inference.

---

## 🌐 Running the Web App Locally

```bash
streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

## 📥 Expected JSON Input Format

The uploaded JSON file must contain historical weather data in chronological order.

Example:

```json
[
  {
    "Datetime": "2026-02-17 10:00:00",
    "Temperature(°C)": 28.5,
    "Radiation(W/m^2)": 450,
    "Cloud_Coverage(%)": 60,
    "Rain(mm/hour)": 0,
    "Relative_Humidity(%)": 72,
    "Wind_Speed(m/s)": 3.5,
    "Pressure(kPa)": 101.2
  }
]
```

Important:
- Datetime must be valid format.
- Data must be sorted or will be auto-sorted.
- The last available timestamp is used as forecasting reference.

---

## 🔮 Prediction Workflow

1. Upload JSON file  
2. Select future date & time  
3. Choose model (XGBoost / Random Forest / Ensemble)  
4. System performs iterative autoregressive forecasting  
5. Physical constraints applied:
   - Radiation = 0 at night
   - Rain cannot be negative
   - Cloud coverage between 0–100%
   - Humidity between 0–100%
   - Wind speed cannot be negative
6. Results displayed with visualization  

---

## 🧪 Model Configuration

### XGBoost
- n_estimators = 200  
- max_depth = 6  
- learning_rate = 0.05  

### Random Forest
- n_estimators = 200  
- max_depth = 12  

### Ensemble Model

```
Final Prediction = 0.6 * XGBoost + 0.4 * RandomForest
```

---

## 📊 Feature Engineering

- Hour, Day, Month, Quarter extraction  
- Cyclical encoding:
  - hour_sin, hour_cos  
  - month_sin, month_cos  
  - day_sin, day_cos  
- Lag features  
- Rolling mean and standard deviation  
- Radiation capping  
- Daytime indicator  
- RobustScaler normalization  

---

## 🌍 Deployment (Streamlit Cloud)

1. Push repository to GitHub  
2. Go to https://streamlit.io/cloud  
3. Connect GitHub repository  
4. Select `app.py`  
5. Deploy  

Public URL will be generated automatically.

---

## 🛠 Tech Stack

- Python  
- Pandas  
- NumPy  
- Scikit-learn  
- XGBoost  
- Streamlit  
- Plotly  

---

## 🎯 Key Highlights

- Multi-output regression (7 weather parameters simultaneously)  
- Physical law enforcement via constraints  
- Modular ML architecture  
- Production-ready artifact saving  
- Clean web-based deployment  

---

## 📌 Author

Built as an end-to-end ML deployment project demonstrating practical machine learning engineering and forecasting system design.
