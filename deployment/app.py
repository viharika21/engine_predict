import streamlit as st
import pandas as pd
from huggingface_hub import hf_hub_download
import joblib

# Download and load the trained model
model_path = hf_hub_download(repo_id="vihu21/predictive_maintenance", filename="best_ada_model.joblib")
model = joblib.load(model_path)

# Streamlit UI
st.title("Engine Condition Prediction")
st.write("""
Fill the engine details below to predict if engine condition good or bad
""")



# User input
Engine_Details = st.number_input("Engine_Details Size (MB)", min_value=1.0, max_value=4000.0, value=50.0, step=0.1)
EngineRpm = st.number_input("Engine rpm", min_value=50, value=3000.0)
LubOilPressure = st.number_input("Lub oil pressure", min_value=0, value=7.25)
FuelPressure = st.number_input("Fuel pressure", min_value=0, value=21.4)
CoolantPressure = st.number_input("Coolant pressure", min_value=0, value=7.5)
lubOilTemp = st.number_input("lub oil temp", min_value=70, value=90.0)
CoolantTemp = st.number_input("Coolant temp", min_value=60, value=195.0)

# ----------------------------
# Prepare input data
# ----------------------------
input_data = pd.DataFrame([{
    'Engine rpm': EngineRpm,
    'Lub oil pressure': LubOilPressure,
    'Fuel pressure': FuelPressure,
    'Coolant pressure': CoolantPressure,
    'lub oil temp': lubOilTemp,
    'Coolant temp': CoolantTemp
}])



# Predict button
if st.button("Predict Engine"):
    prediction = model.predict(input_data)[0]
    st.subheader("Prediction Result:")
    st.success(f"Estimated EngineCondition: **${prediction:,.2f} ")
