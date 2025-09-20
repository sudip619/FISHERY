import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
import joblib
import os
import sys

# --- Configuration ---
# Make sure this matches the name of your main dataset
TRAINING_DATA_FILE = "india_catch_500.csv" 
MODEL_FILE = 'anomaly_model.pkl' # This will overwrite your existing model
ENCODER_FILE = 'encoder.pkl'

print("--- Retraining Anomaly Detection Model ---")

# --- 1. Define the Core Features for the Model ---
# This dictionary maps the column names from your CSV to a standard format.
# The model will ONLY be trained on these features.
COLUMN_MAP = {
    'species_scientific_name': 'species_name',
    'weight_kg': 'weight_kg',
    'latitude': 'latitude',
    'longitude': 'longitude',
    'fishing_method': 'gear_type'
}

# --- 2. Load and Preprocess the Dataset ---
if not os.path.exists(TRAINING_DATA_FILE):
    print(f"Error: Training data file not found at '{TRAINING_DATA_FILE}'")
    sys.exit(1)

print(f"Loading data from '{TRAINING_DATA_FILE}'...")
df = pd.read_csv(TRAINING_DATA_FILE)

# Check if all required source columns exist in the file
source_columns = list(COLUMN_MAP.keys())
if not all(col in df.columns for col in source_columns):
    print(f"Error: Your CSV is missing one of the required columns: {source_columns}")
    sys.exit(1)

# Rename columns to our standard format
df.rename(columns=COLUMN_MAP, inplace=True)

# --- 3. Clean and Prepare Data for AI ---
print("Preparing data for the AI model...")
core_features = list(COLUMN_MAP.values())
df_features = df[core_features].copy()

# Clean numerical data
for col in ['latitude', 'longitude', 'weight_kg']:
    df_features[col] = pd.to_numeric(df_features[col], errors='coerce')

# Drop any rows with missing essential data for training
df_features.dropna(inplace=True)

# Separate features for encoding
numerical_df = df_features[['latitude', 'longitude', 'weight_kg']]
categorical_df = df_features[['species_name', 'gear_type']]

# --- 4. Encode and Train ---
encoder = OneHotEncoder(handle_unknown='ignore')
encoded_cats = encoder.fit_transform(categorical_df)
encoded_df = pd.DataFrame(encoded_cats.toarray())

# Combine final features
df_processed = pd.concat([numerical_df.reset_index(drop=True), encoded_df], axis=1)
df_processed.columns = df_processed.columns.astype(str)

print("Training the Isolation Forest model...")
model = IsolationForest(contamination=0.01, random_state=42)
model.fit(df_processed)

# --- 5. Save the Model and Encoder ---
joblib.dump(model, MODEL_FILE)
joblib.dump(encoder, ENCODER_FILE)

print(f"\n--- Success! ---")
print(f"Model successfully retrained and saved to: {MODEL_FILE}")
print(f"Encoder successfully saved to: {ENCODER_FILE}")