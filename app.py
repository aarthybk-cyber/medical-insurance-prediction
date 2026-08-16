import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression # Explicitly import LinearRegression
import streamlit as st

# Load the saved model and preprocessors
linear_reg_model = joblib.load('linear_reg_model.joblib')
le_sex = joblib.load('label_encoder_sex.joblib')
le_smoker = joblib.load('label_encoder_smoker.joblib')
scaler = joblib.load('standard_scaler.joblib')

st.success("Model and preprocessors loaded successfully!")

def predict_charges(age, sex, bmi, children, smoker, region):
    # Create a DataFrame from the input data
    input_df = pd.DataFrame([[age, sex, bmi, children, smoker, region]],
                            columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])

    # Apply label encoding for 'sex' and 'smoker'
    input_df['sex'] = le_sex.transform(input_df['sex'])
    input_df['smoker'] = le_smoker.transform(input_df['smoker'])

    # Apply one-hot encoding for 'region'
    region_cols = ['region_northwest', 'region_southeast', 'region_southwest']
    for col in region_cols:
        input_df[col] = 0 # Initialize to 0

    if region == 'northwest':
        input_df['region_northwest'] = 1
    elif region == 'southeast':
        input_df['region_southeast'] = 1
    elif region == 'southwest':
        input_df['region_southwest'] = 1

    input_df.drop('region', axis=1, inplace=True)

    # Scale numerical features (age, bmi)
    temp_df_for_scaling = pd.DataFrame(columns=['age', 'bmi', 'charges'])
    temp_df_for_scaling['age'] = input_df['age']
    temp_df_for_scaling['bmi'] = input_df['bmi']
    temp_df_for_scaling['charges'] = 0 # Dummy value, will be ignored for scaling age/bmi

    scaled_numerical_features = scaler.transform(temp_df_for_scaling)
    input_df['age'] = scaled_numerical_features[:, 0]
    input_df['bmi'] = scaled_numerical_features[:, 1]

    # Ensure the order of columns matches the training data used for the model
    final_input = input_df[['age', 'sex', 'bmi', 'children', 'smoker', 'region_northwest', 'region_southeast', 'region_southwest']]

    # Make prediction
    scaled_prediction = linear_reg_model.predict(final_input)

    # Inverse transform the prediction.
    dummy_array = np.array([[0, 0, scaled_prediction[0]]])
    original_scale_prediction = scaler.inverse_transform(dummy_array)[:, 2][0]

    return original_scale_prediction

st.title("Medical Insurance Charges Prediction")
st.write("Predict medical insurance charges based on personal data.")

with st.form("prediction_form"):
    age = st.slider("Age", min_value=18, max_value=100, value=30)
    sex = st.selectbox("Sex", ["male", "female"])
    bmi = st.slider("BMI", min_value=10.0, max_value=60.0, value=25.0)
    children = st.slider("Children", min_value=0, max_value=5, step=1, value=1)
    smoker = st.selectbox("Smoker", ["yes", "no"])
    region = st.selectbox("Region", ["northeast", "northwest", "southeast", "southwest"])

    submitted = st.form_submit_button("Predict Charges")

    if submitted:
        # Convert smoker and sex to 'yes'/'no' for the predict_charges function if needed, 
        # but the selectbox already returns string values.
        # The predict_charges function expects 'sex' and 'smoker' as strings ('male'/'female', 'yes'/'no').
        # No conversion needed here as st.selectbox already provides the correct format.
        
        estimated_charges = predict_charges(age, sex, bmi, children, smoker, region)
        st.success(f"Estimated Medical Charges: ${estimated_charges:.2f}")

print("Streamlit interface defined and ready.")
