# 🌦 Bakkhali Weather Prediction AI

<div align="center">
  
  ![Python](https://img.shields.io/badge/Python-3.8-blue)
  ![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green)
  ![Streamlit](https://img.shields.io/badge/Streamlit-1.28-red)
  ![Docker](https://img.shields.io/badge/Docker-24.0-blue)
  ![XGBoost](https://img.shields.io/badge/XGBoost-2.0-orange)
  ![License](https://img.shields.io/badge/License-MIT-yellow)
  
  **An end-to-end Multi-Output Time Series Weather Forecasting System for Bakkhali Beach, West Bengal**
  
</div>

---


## ✨ Features

### 🤖 Machine Learning
- **Multi-Output Regression**: Predicts 7 weather parameters simultaneously
- **Ensemble Learning**: Combines XGBoost and Random Forest for superior accuracy
- **Physical Constraints**: Automatically applies real-world rules (e.g., radiation = 0 at night)
- **Auto-Retraining**: Weekly automatic model updates with fresh data

### 🌐 API & Backend
- **RESTful API**: Built with FastAPI, fully documented with Swagger
- **Async Support**: Handles multiple requests efficiently
- **Pydantic Validation**: Request/response validation with detailed schemas
- **Background Tasks**: Non-blocking model training

### 🎨 Frontend
- **Interactive Dashboard**: Built with Streamlit
- **Real-time Predictions**: Get forecasts for any future date/time
- **Visual Analytics**: Interactive charts with Plotly
- **Responsive Design**: Works on desktop and mobile

### 🐳 DevOps
- **Docker Support**: Containerized application
- **Docker Compose**: Orchestrate multiple services
- **Health Checks**: Automatic service monitoring
- **Scalable**: Easy to scale horizontally

---

## 📁 Project Structure
weather-prediction-app/
│
├── 📂 backend/
│   ├── __init__.py
│   ├── main.py                 
│   ├── models/
│   │   ├── __init__.py
│   │   ├── predictor.py        
│   │   ├── trainer.py          
│   │   └── schemas.py          
│   ├── data/
│   │   ├── __init__.py
│   │   └── fetcher.py          
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── features.py         
│   │   └── constraints.py      
│   └── scheduler/
│       └── trainer_scheduler.py
│
├── 📂 frontend/
│   ├── app.py                   
│   └── .streamlit/
│       └── config.toml
│
├── 📂 models/                    
│   ├── random_forest_model.pkl
│   ├── xgboost_model.pkl
│   ├── scaler.pkl
│   ├── feature_cols.pkl
│   └── targets.pkl
│
├── 📂 data/                       
│   └── .gitkeep
│
├── 📂 docker/
│   ├── backend.Dockerfile
│   ├── frontend.Dockerfile
│   └── nginx.conf
│
├── docker-compose.yml
├── .env.example
├── .gitignore
├── requirements-backend.txt
├── requirements-frontend.txt
└── README.md


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
Final Prediction = 0.2 * XGBoost + 0.8 * RandomForest
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

### **Data Processing**
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `scikit-learn` - Feature scaling and preprocessing

### **Machine Learning**
- `XGBoost` - Gradient boosting framework
- `Random Forest` - Ensemble learning
- `joblib` - Model serialization

### **Backend**
- `FastAPI` - Modern web framework
- `Pydantic` - Data validation
- `Uvicorn` - ASGI server
- `requests` - HTTP client

### **Frontend**
- `Streamlit` - Web application framework
- `Plotly` - Interactive visualizations
- `pandas` - Data handling

### **DevOps**
- `Docker` - Containerization
- `Docker Compose` - Multi-container orchestration
- `schedule` - Task scheduling

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
