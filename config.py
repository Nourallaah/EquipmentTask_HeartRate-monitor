# config.py
import os

# Project paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DATA_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DATA_DIR = os.path.join(DATA_DIR, "processed")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Create directories if they don't exist
for directory in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    os.makedirs(directory, exist_ok=True)

# Dataset URLs
DATASET_URLS = {
    "chfdb": "https://physionet.org/files/chfdb/1.0.0/",
    "mit_bih": "https://physionet.org/files/mitdb/1.0.0/",
    "fhr": "https://physionet.org/files/ctu-uhb-ctgdb/1.0.0/"
}

# Analysis parameters
SAMPLE_RATE = 1000  # Hz
HRV_WINDOW_SIZE = 300  # seconds for HRV analysis