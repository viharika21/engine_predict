from huggingface_hub import HfApi
import os

# -----------------------------
# CONFIGURATION
# -----------------------------
HF_USERNAME = "vihu21"
DATASET_NAME = "predictive_maintenance"
REPO_ID = f"{HF_USERNAME}/{DATASET_NAME}"

RAW_DATA_PATH = "master/data/engine_data.csv"

# -----------------------------
# CREATE DATASET REPOSITORY
# -----------------------------
api = HfApi()

api.create_repo(
    repo_id=REPO_ID,
    repo_type="dataset",
    private=False,        # set True if required
    exist_ok=True
)

print(f"Dataset repository created: {REPO_ID}")

# -----------------------------
# UPLOAD RAW DATA FILE
# -----------------------------
api.upload_file(
    path_or_fileobj=RAW_DATA_PATH,
    path_in_repo="master/data/engine_data.csv",
    repo_id=REPO_ID,
    repo_type="dataset"
)

print("dataset uploaded successfully.")
