from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib
import os
from typing import List

# --- Configuration ---
MODEL_FILE = 'anomaly_model.pkl'
ENCODER_FILE = 'encoder.pkl'

print("--- Anomaly Detection API (FastAPI) ---")

# --- Initialize the FastAPI App ---
app = FastAPI(title="AquaBase AI API", version="1.0")

# --- CORS Middleware ---
# This allows your website builder to communicate with the API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# --- Load the Pre-trained Model and Encoder ---
print("Loading pre-trained model and encoder...")
if not os.path.exists(MODEL_FILE) or not os.path.exists(ENCODER_FILE):
    print(f"Error: Model or encoder file not found.")
    exit()

model = joblib.load(MODEL_FILE)
encoder = joblib.load(ENCODER_FILE)
print("Model and encoder loaded successfully.")

# --- Define the Data Structures (with Pydantic) ---
# This defines what a single fish catch record should look like in a request.
# FastAPI will use this for automatic validation.
class FishCatch(BaseModel):
    catch_id: int
    species_name: str
    latitude: float
    longitude: float
    catch_weight_kg: float
    gear_type: str
    # You can add other fields from your CSV here if needed

# This defines the structure of the incoming request body.
class AnomalyRequest(BaseModel):
    data: List[FishCatch]

# --- Define the API Endpoint ---
@app.post("/detect-anomalies")
def detect_anomalies(request: AnomalyRequest):
    print(f"\nReceived a request to /detect-anomalies with {len(request.data)} records...")
    
    # Convert Pydantic models to a Pandas DataFrame
    df_input = pd.DataFrame([vars(record) for record in request.data])
    original_ids = df_input['catch_id']

    # Prepare data for the model (identical to training)
    features = ['species_name', 'latitude', 'longitude', 'catch_weight_kg', 'gear_type']
    df_features = df_input[features].copy().fillna({
        'species_name': 'Unknown', 'gear_type': 'Unknown', 'catch_weight_kg': 0
    })

    encoded_cats = encoder.transform(df_features[['species_name', 'gear_type']])
    encoded_df = pd.DataFrame(encoded_cats.toarray())
    numerical_features = df_features[['latitude', 'longitude', 'catch_weight_kg']].reset_index(drop=True)
    
    df_processed = pd.concat([numerical_features, encoded_df], axis=1)
    df_processed.columns = df_processed.columns.astype(str)

    # Use the model to predict anomalies
    predictions = model.predict(df_processed)
    anomaly_ids = [int(original_ids[i]) for i, p in enumerate(predictions) if p == -1]
    
    print(f"Found {len(anomaly_ids)} anomalies.")

    # Return the results
    return {"anomalous_catch_ids": anomaly_ids}

# A simple root endpoint to confirm the API is running
@app.get("/")
def read_root():
    return {"message": "Welcome to the AquaBase AI API"}