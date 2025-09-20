import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import OneHotEncoder
import joblib
import os
import sys

# --- Configuration ---
# The new, richer CSV file you are using for training
TRAINING_DATA_FILE = "india_catch_5000.csv" 
MODEL_FILE = 'anomaly_model.pkl' # We are overwriting the original model
ENCODER_FILE = 'encoder.pkl'

print("--- Training Anomaly Detection Model (Flexible Columns) ---")

# --- 1. Define the Core Features for the Model ---
# These are the only columns the model will ever see.
# This makes our model independent of the source file's naming conventions.
CORE_FEATURES = {
    'numerical': ['latitude', 'longitude', 'weight_kg'],
    'categorical': ['species_name', 'gear_type']
}

# This dictionary maps the column names from your new CSV to our standard names.
COLUMN_MAP = {
    'species_scientific_name': 'species_name',
    'weight_kg': 'weight_kg',
    'latitude': 'latitude',
    'longitude': 'longitude',
    'fishing_method': 'gear_type'
}

# --- 2. Load and Preprocess the Dataset ---
if not os.path.exists(TRAINING_DATA_FILE):
    print(f"Error: Data file not found at '{TRAINING_DATA_FILE}'")
    sys.exit(1)

print(f"Loading data from '{TRAINING_DATA_FILE}'...")
df = pd.read_csv(TRAINING_DATA_FILE)

# Check if all required source columns exist in the file
if not all(col in df.columns for col in COLUMN_MAP.keys()):
    print("Error: Your CSV is missing one of the required columns for the model.")
    sys.exit(1)

# Rename columns to our standard format
df.rename(columns=COLUMN_MAP, inplace=True)

# Select ONLY the core features
all_core_features = CORE_FEATURES['numerical'] + CORE_FEATURES['categorical']
df_features = df[all_core_features].copy()

# --- 3. Clean and Prepare Data for AI ---
print("Preparing data for the AI model...")
# Fill missing numerical values with the column's median
for col in CORE_FEATURES['numerical']:
    df_features[col] = pd.to_numeric(df_features[col], errors='coerce')
df_features.fillna(df_features.median(numeric_only=True), inplace=True)

# Fill missing categorical values with 'Unknown'
for col in CORE_FEATURES['categorical']:
    df_features[col].fillna('Unknown', inplace=True)

# --- 4. Encode and Train ---
encoder = OneHotEncoder(handle_unknown='ignore')
encoded_cats = encoder.fit_transform(df_features[CORE_FEATURES['categorical']])
numerical_df = df_features[CORE_FEATURES['numerical']].reset_index(drop=True)
encoded_df = pd.DataFrame(encoded_cats.toarray())
df_processed = pd.concat([numerical_df, encoded_df], axis=1)
df_processed.columns = df_processed.columns.astype(str)

print("Training the Isolation Forest model...")
model = IsolationForest(contamination=0.01, random_state=42)
model.fit(df_processed)

# --- 5. Save the Model and Encoder ---
joblib.dump(model, MODEL_FILE)
joblib.dump(encoder, ENCODER_FILE)

print(f"\n--- Success! ---")
print(f"Model retrained with selected columns and saved to: {MODEL_FILE}")