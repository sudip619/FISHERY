from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from typing import List, Any, Optional

# --- Configuration ---
MODEL_FILE = 'anomaly_model.pkl'
ENCODER_FILE = 'encoder.pkl'

print("--- Anomaly Detection API (Detects Blanks as Anomalies) ---")

app = FastAPI(title="AquaBase AI API", version="4.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# --- Load Model ---
print("Loading pre-trained model and encoder...")
model = joblib.load(MODEL_FILE)
encoder = joblib.load(ENCODER_FILE)
print("Model and encoder loaded successfully.")

# --- Define Flexible Data Structures ---
class FishCatchRequest(BaseModel):
    catch_id: Any
    species_name: Optional[str] = None
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    weight_kg: Optional[float] = None
    gear_type: Optional[str] = None

class AnomalyRequest(BaseModel):
    data: List[FishCatchRequest]

# --- API Endpoint ---
@app.post("/detect-anomalies")
def detect_anomalies(request: AnomalyRequest):
    print(f"\nReceived request with {len(request.data)} records...")
    df_input = pd.DataFrame([vars(record) for record in request.data])

    # --- Server-Side Data Cleaning & Imputation ---
    df_input['catch_id'] = pd.to_numeric(df_input['catch_id'].astype(str).str.extract(r'(\d+)', expand=False), errors='coerce')
    df_input.dropna(subset=['catch_id'], inplace=True)
    df_input['catch_id'] = df_input['catch_id'].astype(int)

    # --- THIS IS THE FIX ---
    # Impute blank numerical values with a placeholder that the AI will see as an anomaly.
    ANOMALY_PLACEHOLDER = -999.0
    for col in ['latitude', 'longitude', 'weight_kg']:
        df_input[col] = pd.to_numeric(df_input[col], errors='coerce')
        df_input[col].fillna(ANOMALY_PLACEHOLDER, inplace=True)

    # Impute blank categorical values with 'Unknown'
    for col in ['species_name', 'gear_type']:
        df_input[col].fillna('Unknown', inplace=True)

    # --- Prepare Data & Predict ---
    features_for_model = df_input[['species_name', 'latitude', 'longitude', 'weight_kg', 'gear_type']]
    encoded_cats = encoder.transform(features_for_model[['species_name', 'gear_type']])
    df_processed = pd.concat([
        features_for_model[['latitude', 'longitude', 'weight_kg']].reset_index(drop=True),
        pd.DataFrame(encoded_cats.toarray())
    ], axis=1)
    df_processed.columns = df_processed.columns.astype(str)

    predictions = model.predict(df_processed)
    df_input['is_anomaly'] = predictions
    anomaly_ids = df_input[df_input['is_anomaly'] == -1]['catch_id'].tolist()
    
    print(f"Found {len(anomaly_ids)} anomalies after cleaning.")
    return {"anomalous_catch_ids": sorted(anomaly_ids)}