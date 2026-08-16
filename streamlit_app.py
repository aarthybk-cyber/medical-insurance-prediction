# ==========================================================
# Medical Insurance Charges Prediction
# Machine Learning Project
# ==========================================================
!pip install streamlit
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path

# ----------------------------------------------------------
# Page Configuration
# ----------------------------------------------------------
st.set_page_config(
    page_title="Medical Insurance Charges Predictor",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ----------------------------------------------------------
# Custom Styling
# ----------------------------------------------------------
st.markdown("""
<style>

.main-title{
    font-size:38px;
    font-weight:bold;
    color:#0F62FE;
}

.sub-title{
    font-size:18px;
    color:#555555;
}

.footer{
    text-align:center;
    color:gray;
    font-size:14px;
}

</style>
""", unsafe_allow_html=True)

# ----------------------------------------------------------
# Load ML Artifacts
# ----------------------------------------------------------
BASE_DIR = Path(__file__).parent

@st.cache_resource
def load_artifacts():

    model = joblib.load(BASE_DIR / "linear_reg_model.joblib")

    le_sex = joblib.load(
        BASE_DIR / "label_encoder_sex.joblib"
    )

    le_smoker = joblib.load(
        BASE_DIR / "label_encoder_smoker.joblib"
    )

    scaler = joblib.load(
        BASE_DIR / "standard_scaler.joblib"
    )

    return model, le_sex, le_smoker, scaler


try:

    model, le_sex, le_smoker, scaler = load_artifacts()

except Exception as e:

    st.error("Unable to load ML model.")

    st.exception(e)

    st.stop()

# ----------------------------------------------------------
# Sidebar
# ----------------------------------------------------------
with st.sidebar:

    st.image(
        "https://img.icons8.com/color/96/hospital-3.png",
        width=70
    )

    st.title("Medical Insurance Predictor")

    st.markdown("---")

    st.subheader("About")

    st.write(
        """
This application predicts annual
medical insurance charges using a
Machine Learning model trained on
historical insurance data.
"""
    )

    st.markdown("---")

    st.subheader("Model")

    st.success("Linear Regression")

    st.markdown("### Features")

    st.write("✅ Age")
    st.write("✅ Gender")
    st.write("✅ BMI")
    st.write("✅ Children")
    st.write("✅ Smoker")
    st.write("✅ Region")

    st.markdown("---")

    st.subheader("Technologies")

    st.write("• Python")
    st.write("• Streamlit")
    st.write("• Pandas")
    st.write("• NumPy")
    st.write("• Scikit-Learn")

    st.markdown("---")

    st.caption(
        "Developed by Aarthy Balakrishnan"
    )

# ----------------------------------------------------------
# Header
# ----------------------------------------------------------
st.markdown(
    '<p class="main-title">🏥 Medical Insurance Charges Predictor</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-title">Estimate annual medical insurance charges using Machine Learning.</p>',
    unsafe_allow_html=True
)

st.divider()

# ----------------------------------------------------------
# Two Column Layout
# ----------------------------------------------------------
left_col, right_col = st.columns([1.2,1])

# ----------------------------------------------------------
# User Input Form
# ----------------------------------------------------------
with left_col:

    st.subheader("👤 Personal Information")

    with st.form("prediction_form"):

        age = st.slider(
            "Age",
            18,
            100,
            30
        )

        sex = st.selectbox(
            "Gender",
            ["female","male"]
        )

        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=60.0,
            value=25.0,
            step=0.1
        )

        if bmi > 45:

            st.warning(
                "Please verify the BMI value entered."
            )

        children = st.slider(
            "Number of Children",
            0,
            5,
            0
        )

        smoker = st.selectbox(
            "Smoker",
            ["no","yes"]
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

        col1, col2 = st.columns(2)

        with col1:

            submitted = st.form_submit_button(
                "💰 Predict"
            )

        with col2:

            reset = st.form_submit_button(
                "🔄 Reset"
            )

# ----------------------------------------------------------
# Prediction Area
# ----------------------------------------------------------
with right_col:

    st.subheader("📊 Prediction Result")

    prediction_placeholder = st.empty()

    summary_placeholder = st.empty()
    # ----------------------------------------------------------
# Prediction Logic
# ----------------------------------------------------------

USE_INVERSE_SCALER = False
# Change to True ONLY if your model predicts scaled charges.


if submitted:

    try:

        with st.spinner("Predicting insurance charges..."):

            # ------------------------------------------
            # Encode Categorical Variables
            # ------------------------------------------

            sex_encoded = le_sex.transform([sex])[0]

            smoker_encoded = le_smoker.transform([smoker])[0]

            # ------------------------------------------
            # One Hot Encoding for Region
            # ------------------------------------------

            region_northwest = 0
            region_southeast = 0
            region_southwest = 0

            if region == "northwest":
                region_northwest = 1

            elif region == "southeast":
                region_southeast = 1

            elif region == "southwest":
                region_southwest = 1

            # ------------------------------------------
            # Create Input DataFrame
            # ------------------------------------------

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

            # ------------------------------------------
            # Ensure Correct Feature Order
            # ------------------------------------------

            input_df = input_df[
                [
                    "age",
                    "sex",
                    "bmi",
                    "children",
                    "smoker",
                    "region_northwest",
                    "region_southeast",
                    "region_southwest"
                ]
            ]

            # ------------------------------------------
            # Prediction
            # ------------------------------------------

            prediction = model.predict(input_df)[0]

            # ------------------------------------------
            # Optional Inverse Scaling
            # ------------------------------------------

            if USE_INVERSE_SCALER:

                dummy = np.array([
                    [
                        0,
                        0,
                        prediction
                    ]
                ])

                prediction = scaler.inverse_transform(dummy)[0][2]

        # --------------------------------------------------
        # Prediction Card
        # --------------------------------------------------

        prediction_placeholder.success("Prediction Completed Successfully!")

        prediction_placeholder.metric(
            "Estimated Annual Insurance Charges",
            f"${prediction:,.2f}"
        )

        # --------------------------------------------------
        # Prediction Summary
        # --------------------------------------------------

        with summary_placeholder.container():

            st.markdown("---")

            st.subheader("📋 Prediction Summary")

            col1, col2 = st.columns(2)

            with col1:

                st.write(f"**Age:** {age}")

                st.write(f"**Gender:** {sex.title()}")

                st.write(f"**BMI:** {bmi}")

            with col2:

                st.write(f"**Children:** {children}")

                st.write(f"**Smoker:** {smoker.title()}")

                st.write(f"**Region:** {region.title()}")

            st.markdown("---")

            # ------------------------------------------
            # BMI Category
            # ------------------------------------------

            st.subheader("⚕ BMI Category")

            if bmi < 18.5:

                st.info("Underweight")

            elif bmi < 25:

                st.success("Normal Weight")

            elif bmi < 30:

                st.warning("Overweight")

            else:

                st.error("Obese")

            st.markdown("---")

            # ------------------------------------------
            # Health Tips
            # ------------------------------------------

            st.subheader("💡 Health Tips")

            if smoker == "yes":

                st.warning(
                    "Smoking significantly increases medical insurance costs."
                )

            else:

                st.success(
                    "Non-smokers generally pay lower insurance premiums."
                )

            if bmi >= 30:

                st.info(
                    "Maintaining a healthy BMI can help reduce future health risks."
                )

            st.markdown("---")

            st.caption(
                "This prediction is generated using a Machine Learning model "
                "and should be treated as an estimate."
            )

            # ------------------------------------------
            # Debug Mode
            # ------------------------------------------

            with st.expander("🔍 Show Encoded Features"):

                st.dataframe(input_df)

    except Exception as e:

        st.error("Prediction Failed")

        st.exception(e)

# ----------------------------------------------------------
# Footer
# ----------------------------------------------------------

st.divider()

st.caption(
    "🏥 Medical Insurance Charges Predictor | "
    "Machine Learning Project  "
    
)
