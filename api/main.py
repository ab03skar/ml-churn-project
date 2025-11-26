from pathlib import Path
import json

import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from joblib import load


root = Path(__file__).parents[1]

model_path = root / "models" / "churn_model.joblib"
feature_columns_path = root / "models" / "feature_columns.json"

model = load(model_path)

with open(feature_columns_path, "r") as f:
    feature_columns = json.load(f)


class ChurnFeatures(BaseModel):
    num_events: int
    num_sessions: int
    num_songs: int
    num_ads: int
    thumbs_up: int
    thumbs_down: int
    help_events: int
    error_events: int
    downgrade_events: int
    distinct_artists: int
    distinct_songs: int
    is_paid: int
    days_on_platform: int
    activity_span_days: int
    songs_per_session: float
    ads_per_session: float
    thumbs_up_ratio: float


app = FastAPI(title="Churn Prediction API")


@app.get("/")
def read_root():
    return {"message": "Churn prediction API is running"}


@app.post("/predict")
def predict_churn(features: ChurnFeatures):
    data_row = [getattr(features, col) for col in feature_columns]
    X = pd.DataFrame([data_row], columns=feature_columns)

    pred = model.predict(X)[0]
    prob = float(model.predict_proba(X)[0][1])

    return {
        "churn_prediction": int(pred),
        "churn_probability": prob,
    }
