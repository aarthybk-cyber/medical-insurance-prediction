import streamlit as st
import pandas as pd
import joblib

# Load model and preprocessors
model = joblib.load("linear_reg_model.joblib")
le_sex = joblib.load("label_encoder_sex.joblib")
le_smoker = joblib.load("label_encoder_smoker.joblib")

st.set_page_config(
    page_title="Medical Insurance Charges Predictor",
    page_icon="🏥",
    layout="centered"
)

st.title("🏥 Medical Insurance Charges Prediction")

st.write("Predict estimated medical insurance charges using a trained Linear Regression model.")

age = st.slider("Age", 18, 100, 30)

sex = st.selectbox(
    "Sex",
    ["male", "female"]
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
    ["no", "yes"]
)

region = st.selectbox(
    "Region",
    ["northeast", "northwest", "southeast", "southwest"]
)

if st.button("Predict Charges"):

    sex = le_sex.transform([sex])[0]
    smoker = le_smoker.transform([smoker])[0]

    input_df = pd.DataFrame({
        "age": [age],
        "sex": [sex],
        "bmi": [bmi],
        "children": [children],
        "smoker": [smoker],
        "region_northwest": [0],
        "region_southeast": [0],
        "region_southwest": [0]
    })

    if region == "northwest":
        input_df["region_northwest"] = 1
    elif region == "southeast":
        input_df["region_southeast"] = 1
    elif region == "southwest":
        input_df["region_southwest"] = 1

    prediction = model.predict(input_df)

    st.success(f"Estimated Insurance Charges: ${prediction[0]:,.2f}")
