import requests
import pandas as pd
import sys
import numpy as np

# --- Configuration ---
API_URL = "http://127.0.0.1:8000/detect-anomalies"
TEST_DATA_FILE = "india_catch_5000.csv"

print("--- AquaBase API Test Client (Final JSON-Safe Version) ---")

# --- 1. Define Column Mapping ---
COLUMN_MAP = {
    'species_scientific_name': 'species_name',
    'weight_kg': 'weight_kg',
    'latitude': 'latitude',
    'longitude': 'longitude',
    'fishing_method': 'gear_type'
}
API_REQUIRED_COLUMNS = list(COLUMN_MAP.keys()) + ['catch_id']

# --- 2. Load and Preprocess Data ---
print(f"Reading test data from '{TEST_DATA_FILE}'...")
try:
    df_full = pd.read_csv(TEST_DATA_FILE)
except FileNotFoundError:
    print(f"--- ERROR: File not found: '{TEST_DATA_FILE}' ---")
    sys.exit(1)

if not all(col in df_full.columns for col in API_REQUIRED_COLUMNS):
    print(f"--- ERROR: Your CSV is missing required columns: {API_REQUIRED_COLUMNS} ---")
    sys.exit(1)

df_api_input = df_full[API_REQUIRED_COLUMNS].copy()
df_api_input.rename(columns=COLUMN_MAP, inplace=True)

# --- 3. THE CRUCIAL FIX: Make Data JSON-Safe ---
# Replace pandas' NaN with Python's None, which converts to 'null' in JSON.
df_processed = df_api_input.replace({np.nan: None})

# --- 4. Send Request to the API ---
data_to_send = df_processed.to_dict(orient='records')
request_payload = {"data": data_to_send}
print(f"Prepared {len(data_to_send)} JSON-safe records to send.")

try:
    response = requests.post(API_URL, json=request_payload)
    response.raise_for_status()
except requests.exceptions.RequestException as e:
    print(f"\n--- API ERROR: {e} ---")
    print("Please check your FastAPI server terminal for detailed errors.")
    sys.exit(1)

# --- 5. Print the Results ---
print("\n--- API Response Received ---")
result = response.json()
anomalous_ids = result.get('anomalous_catch_ids', [])

print(f"\n--- Test Result ---")
if not anomalous_ids:
    print("ℹ️  The API did not flag any records as anomalies.")
else:
    print(f"✅ SUCCESS: The API identified {len(anomalous_ids)} record(s) as anomalies.")
    print("The following catch_ids were flagged: " + ", ".join(map(str, sorted(anomalous_ids))))