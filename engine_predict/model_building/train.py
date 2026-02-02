import pandas as pd
import joblib
from pyngrok import ngrok
import subprocess
import mlflow
from datasets import load_dataset
from huggingface_hub import hf_hub_download
import mlflow.sklearn
from sklearn.metrics import classification_report

# Set your auth token here (replace with your actual token)
ngrok.set_auth_token("35sZZXdVWGqR8CTDSZ4Ig2g3plm_FW9eq7c9TFgtsKqrHugW")

# Start MLflow UI on port 5000
process = subprocess.Popen(["mlflow", "ui", "--port", "5000"])

# Create public tunnel
public_url = ngrok.connect(5000).public_url
print("MLflow UI is available at:", public_url)

# Set the tracking URL for MLflow
mlflow.set_tracking_uri(public_url)

# Set the name for the experiment
mlflow.set_experiment("AdaBoost-Predictive-Maintenance")

HF_DATASET = "vihu21/predictive_maintenance"
target_col = 'Engine Condition'
# Load Xtest
dataset3 = load_dataset(HF_DATASET, data_files="engine_predict/master/data/Xtest.csv")
X_test = pd.DataFrame(dataset3['train'])

# Load ytest
dataset4 = load_dataset(HF_DATASET, data_files="engine_predict/master/data/ytest.csv")
test_dataset = dataset4['train']
df_ytest = pd.DataFrame(list(test_dataset[target_col]), columns=[target_col])
y_test= df_ytest[target_col].values.ravel()

# Load model from Hugging Face

MODEL_FILE = "engine_predict/best_ada_model.joblib"   # ⚠️ make sure this matches HF exactly

model_path = hf_hub_download(
    repo_id=HF_DATASET,
    filename=MODEL_FILE,
    repo_type="model"
)

ada_model = joblib.load(model_path)
print("✅ Model loaded from Hugging Face")


with mlflow.start_run(run_name="AdaBoost-from-HF"):
    y_pred = ada_model.predict(X_test)
    report = classification_report(y_test, y_pred, output_dict=True)

    mlflow.log_metrics({
        "accuracy": report["accuracy"],
        "precision": report["1"]["precision"],
        "recall": report["1"]["recall"],
        "f1_score": report["1"]["f1-score"]
    })

    mlflow.log_params(ada_model.get_params())

    mlflow.sklearn.log_model(
        ada_model,
        artifact_path="model",
        registered_model_name="AdaBoostPredictiveMaintenance"
    )
