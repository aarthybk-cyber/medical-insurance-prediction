import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression # Explicitly import LinearRegression
import gradio as gr

# Load the saved model and preprocessors
linear_reg_model = joblib.load('linear_reg_model.joblib')
le_sex = joblib.load('label_encoder_sex.joblib')
le_smoker = joblib.load('label_encoder_smoker.joblib')
scaler = joblib.load('standard_scaler.joblib')

print("Model and preprocessors loaded successfully!")

def predict_charges(age, sex, bmi, children, smoker, region):
    # Create a DataFrame from the input data
    input_df = pd.DataFrame([[age, sex, bmi, children, smoker, region]],
                            columns=['age', 'sex', 'bmi', 'children', 'smoker', 'region'])

    # Apply label encoding for 'sex' and 'smoker'
    input_df['sex'] = le_sex.transform(input_df['sex'])
    input_df['smoker'] = le_smoker.transform(input_df['smoker'])

    # Apply one-hot encoding for 'region'
    # Ensure all region columns are present, even if not in current input_df
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
    # The scaler was fitted on ['age', 'bmi', 'charges']
    # We need to scale 'age' and 'bmi' specifically. Create a dummy dataframe for scaling.
    # The order of features in X_train was 'age', 'sex', 'bmi', 'children', 'smoker', 'region_northwest', 'region_southeast', 'region_southwest'
    # We need to create a temporary dataframe with 'age', 'bmi', and a dummy 'charges' column for scaling to work correctly.

    temp_df_for_scaling = pd.DataFrame(columns=['age', 'bmi', 'charges'])
    temp_df_for_scaling['age'] = input_df['age']
    temp_df_for_scaling['bmi'] = input_df['bmi']
    temp_df_for_scaling['charges'] = 0 # Dummy value, will be ignored for scaling age/bmi

    scaled_numerical_features = scaler.transform(temp_df_for_scaling)
    input_df['age'] = scaled_numerical_features[:, 0]
    input_df['bmi'] = scaled_numerical_features[:, 1]

    # Ensure the order of columns matches the training data used for the model
    # The columns should be: 'age', 'sex', 'bmi', 'children', 'smoker', 'region_northwest', 'region_southeast', 'region_southwest'
    # Assuming X_train columns were in this order. If not, explicitly reorder input_df.
    # For this example, let's assume the columns in input_df are already correctly ordered
    # after the above transformations to match X_train, EXCEPT for `charges` which is not a feature

    # The model was trained with 8 features:
    # 'age', 'sex', 'bmi', 'children', 'smoker', 'region_northwest', 'region_southeast', 'region_southwest'
    final_input = input_df[['age', 'sex', 'bmi', 'children', 'smoker', 'region_northwest', 'region_southeast', 'region_southwest']]

    # Make prediction
    scaled_prediction = linear_reg_model.predict(final_input)

    # Inverse transform the prediction. The scaler was fitted on ['age', 'bmi', 'charges'].
    # To inverse transform only 'charges', we need to create a dummy array with 0s for 'age' and 'bmi'.
    dummy_array = np.array([[0, 0, scaled_prediction[0]]])
    original_scale_prediction = scaler.inverse_transform(dummy_array)[:, 2][0]

    return original_scale_prediction

print("Prediction function defined.")

import os

# Set up Gradio Interface
inputs = [
    gr.Slider(minimum=18, maximum=100, label="Age"),
    gr.Dropdown(["male", "female"], label="Sex"),
    gr.Slider(minimum=10, maximum=60, label="BMI"),
    gr.Slider(minimum=0, maximum=5, step=1, label="Children"),
    gr.Dropdown(["yes", "no"], label="Smoker"),
    gr.Dropdown(["northeast", "northwest", "southeast", "southwest"], label="Region")
]

outputs = gr.Number(label="Estimated Medical Charges")

gradio_app = gr.Interface(fn=predict_charges, inputs=inputs, outputs=outputs,
                         title="Medical Insurance Charges Prediction",
                         description="Predict medical insurance charges based on personal data.")


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    gradio_app.launch(server_name="0.0.0.0", server_port=port) # Launch on 0.0.0.0 for external access in Colab

print("Gradio interface defined and ready to launch.")
