import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
import os

# ------------------- TRAIN MODEL -------------------
def train_model():
    df = pd.read_csv("workwave_occupancy_data.csv")
    X = df.drop("Occupancy_Rate", axis=1)
    y = df["Occupancy_Rate"]
    model = RandomForestRegressor(n_estimators=150, random_state=42)
    model.fit(X, y)
    joblib.dump(model, "workwave_model.pkl")
    return model

# Load model or train if not found
if os.path.exists("workwave_model.pkl"):
    model = joblib.load("workwave_model.pkl")
else:
    model = train_model()

# ------------------- FASTAPI APP -------------------
app = FastAPI(title="WorkWave Occupancy Predictor")

# Serve static files like CSS
app.mount("/static", StaticFiles(directory="static"), name="static")


# Serve HTML homepage
@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html", "r", encoding="utf-8") as f:
        return f.read()


# Prediction API
@app.get("/predict", response_class=HTMLResponse)
def predict(
    city: str = Query(...),
    gdp_growth: float = Query(...),
    it_pop: float = Query(...),
    competitor_density: float = Query(...),
    population_density: float = Query(...),
    startup_score: float = Query(...)
):
    input_data = np.array([[gdp_growth, it_pop, competitor_density, population_density, startup_score]])
    prediction = model.predict(input_data)[0]
    percent = min(max(prediction * 3, 0), 100)

    with open("index.html", "r", encoding="utf-8") as f:
        html = f.read()

    return HTMLResponse(content=html + f"<script>updatePrediction({percent});</script>")


# ------------------- PRODUCTION RUN CONFIG -------------------
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
