import streamlit as st
import pandas as pd
import numpy as np
import joblib

# -------------------------------------------------
# Page Configuration
# -------------------------------------------------
st.set_page_config(
    page_title="Medical Insurance Charges Predictor",
    page_icon="🏥",
    layout="centered"
)

# -------------------------------------------------
# Load Model & Encoders
# -------------------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load("linear_reg_model.joblib")
    le_sex = joblib.load("label_encoder_sex.joblib")
    le_smoker = joblib.load("label_encoder_smoker.joblib")
    scaler = joblib.load("standard_scaler.joblib")
    return model, le_sex, le_smoker, scaler


model, le_sex, le_smoker, scaler = load_artifacts()

# -------------------------------------------------
# Sidebar
# -------------------------------------------------
st.sidebar.title("🏥 About")

st.sidebar.info(
    """
This application predicts **Medical Insurance Charges**
using a **Linear Regression Machine Learning Model**.

### Features Used
- Age
- Gender
- BMI
- Children
- Smoker
- Region

### Model
Linear Regression

Developed using **Python**, **Scikit-Learn**
and **Streamlit**.
"""
)

# -------------------------------------------------
# Header
# -------------------------------------------------
st.title("🏥 Medical Insurance Charges Predictor")

st.markdown(
"""
Estimate annual medical insurance charges based on
personal and demographic information.
"""
)

st.divider()

# -------------------------------------------------
# User Input Form
# -------------------------------------------------
with st.form("prediction_form"):

    st.subheader("👤 Personal Details")

    age = st.slider(
        "Age",
        min_value=18,
        max_value=100,
        value=30
    )

    sex = st.selectbox(
        "Gender",
        ["female", "male"]
    )

    bmi = st.number_input(
        "BMI",
        min_value=10.0,
        max_value=60.0,
        value=25.0,
        step=0.1
    )

    children = st.slider(
        "Number of Children",
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
        [
            "northeast",
            "northwest",
            "southeast",
            "southwest"
        ]
    )

    submitted = st.form_submit_button(
        "💰 Predict Insurance Charges"
    )

# -------------------------------------------------
# Prediction
# -------------------------------------------------
if submitted:

    try:

        with st.spinner("Predicting..."):

            sex_encoded = le_sex.transform([sex])[0]
            smoker_encoded = le_smoker.transform([smoker])[0]

            region_northwest = 0
            region_southeast = 0
            region_southwest = 0

            if region == "northwest":
                region_northwest = 1

            elif region == "southeast":
                region_southeast = 1

            elif region == "southwest":
                region_southwest = 1

            input_df = pd.DataFrame({

                "age":[age],

                "sex":[sex_encoded],

                "bmi":[bmi],

                "children":[children],

                "smoker":[smoker_encoded],

                "region_northwest":[region_northwest],

                "region_southeast":[region_southeast],

                "region_southwest":[region_southwest]

            })

            prediction_scaled = model.predict(input_df)

            dummy = np.array([
                [
                    0,
                    0,
                    prediction_scaled[0]
                ]
            ])

            prediction = scaler.inverse_transform(dummy)[0][2]

        st.success("Prediction Successful!")

        st.divider()

        st.subheader("💰 Estimated Insurance Charges")

        st.metric(
            label="Estimated Annual Charges",
            value=f"${prediction:,.2f}"
        )

        st.divider()

        st.subheader("📋 Prediction Summary")

        summary = pd.DataFrame({

            "Feature":[
                "Age",
                "Gender",
                "BMI",
                "Children",
                "Smoker",
                "Region"
            ],

            "Value":[
                age,
                sex.title(),
                bmi,
                children,
                smoker.title(),
                region.title()
            ]

        })

        st.table(summary)

        if bmi < 18.5:
            st.info("BMI Category: Underweight")

        elif bmi < 25:
            st.success("BMI Category: Normal")

        elif bmi < 30:
            st.warning("BMI Category: Overweight")

        else:
            st.error("BMI Category: Obese")

    except Exception as e:

        st.error("Prediction failed.")

        st.exception(e)

# -------------------------------------------------
# Footer
# -------------------------------------------------
st.divider()

st.caption(
    "Medical Insurance Charges Prediction using Linear Regression | Built with Streamlit"
)
