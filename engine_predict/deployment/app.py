# %%writefile app.py
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download
from sklearn.ensemble import AdaBoostClassifier  # ensure scikit-learn flavor
from sklearn.metrics import classification_report

# -------------------------------
# Streamlit App Title & Description
# -------------------------------
st.set_page_config(page_title="Engine Condition Predictor", layout="centered")
st.title("Engine Condition Prediction")
st.write("""
Predict if an engine's condition is Good or Bad based on sensor readings.
""")

# -------------------------------
# Load the model from Hugging Face
# -------------------------------

@st.cache_resource(show_spinner=True)
def load_model():
    """
    Load the pre-trained AdaBoost model from Hugging Face.
    Uses caching to prevent re-downloading every restart.
    """
    HF_DATASET = "vihu21/predictive_maintenance"

    MODEL_FILE = "engine_predict/best_ada_model.joblib"   # ⚠️ make sure this matches HF exactly

    model_path = hf_hub_download(
      repo_id=HF_DATASET,
      filename=MODEL_FILE,
      repo_type="model"
    )
    return joblib.load(model_path)



ada_model = load_model()
st.success("✅ Model loaded successfully!")

# -------------------------------
# User Input
# -------------------------------
st.subheader("Engine Sensor Inputs")
Engine_Details = st.number_input("Engine Details Size (MB)", min_value=1.0, max_value=4000.0, value=50.0, step=0.1)
EngineRpm = st.number_input("Engine rpm", min_value=50, max_value=10000, value=3000)
LubOilPressure = st.number_input("Lubrication Oil Pressure", min_value=0.0, value=7.25)
FuelPressure = st.number_input("Fuel Pressure", min_value=0.0, value=21.4)
CoolantPressure = st.number_input("Coolant Pressure", min_value=0.0, value=7.5)
LubOilTemp = st.number_input("Lubrication Oil Temperature", min_value=0.0, value=90.0)
CoolantTemp = st.number_input("Coolant Temperature", min_value=0.0, value=195.0)

# -------------------------------
# Prepare input dataframe
# -------------------------------
input_data = pd.DataFrame([{
    'Engine rpm': EngineRpm,
    'Lub oil pressure': LubOilPressure,
    'Fuel pressure': FuelPressure,
    'Coolant pressure': CoolantPressure,
    'lub oil temp': LubOilTemp,
    'Coolant temp': CoolantTemp
}])

# -------------------------------
# Make Prediction
# -------------------------------
if st.button("Predict Engine Condition"):
    try:
        prediction = ada_model.predict(input_data)[0]
        
        # Map numeric prediction to labels
        if prediction == 0:
            result = "Good ✅"
        elif prediction == 1:
            result = "Faulty ❌"
        else:
            result = f"Unknown ({prediction})"

        st.subheader("Prediction Result")
        st.success(f"Estimated Engine Condition: **{result}**")
        
    except Exception as e:
        st.error(f"❌ Error during prediction: {str(e)}")
