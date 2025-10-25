import uvicorn
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import joblib
from pyngrok import ngrok
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

# Load or train the model once
if os.path.exists("workwave_model.pkl"):
    model = joblib.load("workwave_model.pkl")
else:
    model = train_model()

# ------------------- FASTAPI APP -------------------
app = FastAPI(title="WorkWave Occupancy Predictor")

# Mount static folder for CSS
app.mount("/static", StaticFiles(directory="static"), name="static")

# Serve index.html
@app.get("/", response_class=HTMLResponse)
def home():
    with open("index.html", "r", encoding="utf-8") as f:
        html_content = f.read()
    return HTMLResponse(content=html_content)

# Prediction endpoint
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
    percent = prediction * 3

    # Return the same page and auto update the bar through JS
    with open("index.html", "r", encoding="utf-8") as f:
        html = f.read()

    return HTMLResponse(
        content=html + f"<script>updatePrediction({percent});</script>"
    )

# ------------------- RUN WITH NGROK -------------------
if __name__ == "__main__":
    ngrok.set_auth_token("34WEvm8lAozqQfF5Hu3NhOMqA2E_2L5gjGFHmfAkqdHP8EeLY")  # Replace with your token

    public_url = ngrok.connect(8000)
    print(f" * Ngrok URL: {public_url}")

    uvicorn.run(app, host="0.0.0.0", port=8000)
