# Medical Insurance Charges Prediction

## Project Overview

This project deploys a machine learning model to predict medical insurance charges based on various personal and demographic factors. The model is a Linear Regression model trained on a dataset containing age, sex, BMI, number of children, smoker status, and region.

The goal of this project is to provide an interactive tool for users to estimate their medical insurance charges. The solution involves:

1.  **Data Cleaning and Preprocessing**: Handling duplicates, encoding categorical features (Label Encoding for 'sex' and 'smoker', One-Hot Encoding for 'region'), and scaling numerical features ('age', 'bmi', 'charges') using `StandardScaler`.
2.  **Model Training**: A Linear Regression model is trained on the preprocessed data.
3.  **Model Evaluation**: The model's performance is evaluated using metrics such as R-squared, MAE, MSE, and RMSE, and through residual analysis.
4.  **Deployment**: The trained model and preprocessors are saved and deployed as a web application using Gradio on Render.

## Technologies Used

- Python
- Pandas
- Scikit-learn
- Gradio
- Joblib
- Render

## Project Structure

-   `app.py`: The main Python script that loads the model and preprocessors, handles input data preprocessing, makes predictions, and defines the Gradio web interface.
-   `requirements.txt`: Lists all the Python libraries and their exact versions required to run the `app.py` script. This ensures a consistent environment on Render.
-   `linear_reg_model.joblib`: The saved Linear Regression model.
-   `label_encoder_sex.joblib`: The saved LabelEncoder for the 'sex' feature.
-   `label_encoder_smoker.joblib`: The saved LabelEncoder for the 'smoker' feature.
-   `standard_scaler.joblib`: The saved StandardScaler, used for scaling 'age' and 'bmi' and inverse-transforming the predicted 'charges'.
-   `README.md`: This file, providing an overview of the project and deployment instructions.

## How the Model Works

This project utilizes a Linear Regression model to predict medical insurance charges. The model was trained on a dataset including various features like age, sex, BMI, number of children, smoker status, and region. Before training, categorical features were encoded and numerical features were scaled to ensure optimal model performance. The deployed application preprocesses new input data in the same way, makes a prediction, and then inverse transforms the predicted charges to provide an estimate in the original currency.

## How to Run Locally

To run the Gradio application locally before deployment:

1.  Ensure you have all the required libraries installed (from `requirements.txt`).
2.  Place `app.py`, `linear_reg_model.joblib`, `label_encoder_sex.joblib`, `label_encoder_smoker.joblib`, and `standard_scaler.joblib` in the same directory.
3.  Run `python app.py` in your terminal.
4.  Open your web browser and go to the address displayed in the terminal (usually `http://127.0.0.1:7860`).

## Deployment

To deploy this project on Render, follow these steps:

1.  **Prepare Files**: Ensure you have `app.py`, `requirements.txt`, `linear_reg_model.joblib`, `label_encoder_sex.joblib`, `label_encoder_smoker.joblib`, `standard_scaler.joblib`, and `README.md` in your project root.
2.  **Create a Git Repository**: Push all these files to a Git repository (e.g., GitHub, GitLab).
3.  **Create a New Web Service on Render**: Log in to your Render account and create a new Web Service. Connect your Git repository.
4.  **Configure Build & Start Commands**: 
    -   Build Command: `pip install -r requirements.txt`
    -   Start Command: `python app.py`
5.  **Environment Variables**: Render automatically sets the `PORT` environment variable, which `app.py` is configured to use.
6.  **Deploy**: Initiate the deployment. Render will build your service and deploy it.

## Sample Prediction

Here's an example of how to use the Gradio application to make a prediction:

-   **Age**: 30
-   **Sex**: Female
-   **BMI**: 25
-   **Children**: 1
-   **Smoker**: No
-   **Region**: Southeast

Upon entering these values into the Gradio interface, the model would output an estimated medical charge, for instance, `$3500.00`. This value represents the predicted annual medical insurance cost for an individual fitting these characteristics.

## Live Demo

[Render service link (once deployed)]

## Author

[Your Name or GitHub Profile]
