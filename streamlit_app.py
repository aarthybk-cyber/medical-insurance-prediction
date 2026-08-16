import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load trained model and preprocessors
model = joblib.load("linear_reg_model.joblib")
sex_encoder = joblib.load("label_encoder_sex.joblib")
smoker_encoder = joblib.load("label_encoder_smoker.joblib")
scaler = joblib.load("standard_scaler.joblib")


st.set_page_config(
    page_title="Medical Insurance Charges Predictor",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Medical Insurance Charges Prediction")
st.write("Predict medical insurance charges based on personal information.")

age = st.slider("Age", 18, 100, 30)

sex = st.selectbox(
    "Sex",
    ["Male", "Female"]
)

bmi = st.number_input(
    "BMI",
    min_value=10.0,
    max_value=60.0,
    value=25.0
)

children = st.slider(
    "Children",
    0,
    5,
    0
)

smoker = st.selectbox(
    "Smoker",
    ["No", "Yes"]
)

region = st.selectbox(
    "Region",
    ["northeast","northwest","southeast","southwest"]
)

if st.button("Predict Charges"):

    sex_encoded = sex_encoder.transform([sex])[0]
    smoker_encoded = smoker_encoder.transform([smoker])[0]

    input_df = pd.DataFrame({
        "age":[age],
        "sex":[sex_encoded],
        "bmi":[bmi],
        "children":[children],
        "smoker":[smoker_encoded],
        "region":[region]
    })

    prediction = model.predict(input_df)

    st.success(f"Estimated Medical Insurance Charges: ${prediction[0]:,.2f}")
