import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Removed st.set_page_config() as it's handled by the main app

# --- Load the trained model, scaler, and encoder ---
try:
    model_path = 'liver_disease_model.pkl'
    scaler_path = 'liver_disease_scaler.pkl'
    encoder_path = 'liver_gender_encoder.pkl'

    if not os.path.exists(model_path):
        st.error(f"Error: Model file '{model_path}' not found. Please ensure 'liver_model_training.py' was run successfully.")
        st.stop()
    if not os.path.exists(scaler_path):
        st.error(f"Error: Scaler file '{scaler_path}' not found. Please ensure 'liver_model_training.py' was run successfully.")
        st.stop()
    if not os.path.exists(encoder_path):
        st.error(f"Error: Gender Encoder file '{encoder_path}' not found. Please ensure 'liver_model_training.py' was run successfully.")
        st.stop()

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    gender_encoder = joblib.load(encoder_path)
    st.sidebar.success("Liver Model, scaler, and gender encoder loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading Liver model, scaler, or encoder: {e}")
    st.stop()

# --- Load Model Performance Data ---
model_performance_df = pd.DataFrame()
performance_chart_path = os.path.join('liver_eda_plots', 'liver_model_performance_comparison_chart.png')
try:
    performance_csv_path = 'liver_model_performance.csv'
    if os.path.exists(performance_csv_path):
        model_performance_df = pd.read_csv(performance_csv_path)
        st.sidebar.success("Liver Model performance data loaded successfully.")
    else:
        st.sidebar.warning("Liver Model performance comparison file 'liver_model_performance.csv' not found. Please run 'liver_model_training.py' to generate it.")
except Exception as e:
    st.sidebar.error(f"Error loading Liver model performance data: {e}")


# --- Title and Description ---
st.title("🩺 Liver Disease Prediction")
st.markdown("""
    This page allows you to predict the likelihood of Liver Disease based on various
    patient biochemical features.
""")

# --- Display Model Performance Comparison ---
if not model_performance_df.empty:
    st.subheader("📊 Model Performance Comparison")
    st.markdown("Here's how different models performed on the Liver Disease dataset:")

    best_model_row = model_performance_df.loc[model_performance_df['F1-Score'].idxmax()] # Using F1-Score for best model selection
    best_model_name = best_model_row['Model']

    st.write(f"The best performing model based on F1-Score is **{best_model_name}**.")

    def highlight_best_model(s):
        is_best = s['Model'] == best_model_name
        return ['background-color: lightgreen' if is_best else '' for _ in s]

    st.dataframe(model_performance_df.style.apply(highlight_best_model, axis=1), hide_index=True)

    st.subheader("📈 Combined Model Performance Comparison Chart")
    if os.path.exists(performance_chart_path):
        st.image(performance_chart_path, caption='Comparison of Model Performance Metrics', use_column_width=True)
        st.write(f"You can find this chart saved at `{performance_chart_path}`.")
    else:
        st.warning(f"Combined model performance chart not found at `{performance_chart_path}`. Please run 'liver_model_training.py' to generate it.")
else:
    st.warning("Liver Model performance data is not available. Please run the training script.")

st.markdown("---")

# --- Input Features ---
st.header("📝 Enter Patient Biochemical Parameters for Prediction")
st.markdown("Please fill in the details below to get a prediction:")

# Define the feature columns in the exact order expected by the model
feature_columns_order = [
    'Age', 'Gender', 'Total_Bilirubin', 'Direct_Bilirubin',
    'Alkaline_Phosphotase', 'Alamine_Aminotransferase',
    'Aspartate_Aminotransferase', 'Total_Protiens', 'Albumin',
    'Albumin_and_Globulin_Ratio'
]

# Create input fields for each feature
input_values = {}

cols_input = st.columns(2)

with cols_input[0]:
    input_values['Age'] = st.number_input('Age', min_value=0, max_value=120, value=40)
    selected_gender = st.radio('Gender', ['Male', 'Female'])
    # Convert gender to encoded numerical value using the loaded encoder
    input_values['Gender'] = gender_encoder.transform([selected_gender])[0]
    input_values['Total_Bilirubin'] = st.number_input('Total Bilirubin (mg/dL)', min_value=0.0, max_value=75.0, value=1.0, step=0.1, format="%.2f")
    input_values['Direct_Bilirubin'] = st.number_input('Direct Bilirubin (mg/dL)', min_value=0.0, max_value=40.0, value=0.2, step=0.1, format="%.2f")
    input_values['Alkaline_Phosphotase'] = st.number_input('Alkaline Phosphotase (IU/L)', min_value=0, max_value=2000, value=150)

with cols_input[1]:
    input_values['Alamine_Aminotransferase'] = st.number_input('Alamine Aminotransferase (SGPT) (IU/L)', min_value=0, max_value=2000, value=25)
    input_values['Aspartate_Aminotransferase'] = st.number_input('Aspartate Aminotransferase (SGOT) (IU/L)', min_value=0, max_value=2000, value=25)
    input_values['Total_Protiens'] = st.number_input('Total Protiens (g/dL)', min_value=0.0, max_value=10.0, value=7.0, step=0.1, format="%.2f") # Corrected key
    input_values['Albumin'] = st.number_input('Albumin (g/dL)', min_value=0.0, max_value=6.0, value=3.5, step=0.1, format="%.2f")
    input_values['Albumin_and_Globulin_Ratio'] = st.number_input('Albumin and Globulin Ratio', min_value=0.0, max_value=3.0, value=1.0, step=0.01, format="%.2f")


# --- Prediction ---
st.markdown("---")
if st.button("Predict Liver Disease"):
    try:
        # Create a DataFrame from input values, ensuring correct column order
        input_df = pd.DataFrame([input_values], columns=feature_columns_order)

        # Scale the input data
        scaled_input_data = scaler.transform(input_df)

        prediction_proba = model.predict_proba(scaled_input_data)[:, 1] # Probability of Liver Disease (class 1)
        prediction = model.predict(scaled_input_data)[0]

        st.subheader("Prediction Result:")
        if prediction == 1:
            st.error(f"Based on the provided information, the model predicts a **HIGH likelihood of Liver Disease**.")
            st.write(f"Confidence (Probability of Disease): **{prediction_proba[0]:.2f}**")
        else:
            st.success(f"Based on the provided information, the model predicts a **LOW likelihood of Liver Disease**.")
            st.write(f"Confidence (Probability of Disease): **{prediction_proba[0]:.2f}**") # Still show probability of disease

        st.markdown("""
            **Disclaimer:** This prediction is based on a machine learning model and
            should not be considered a substitute for professional medical advice.
            Always consult with a qualified healthcare provider for diagnosis and treatment.
        """)
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and Scikit-learn.")