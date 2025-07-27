import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

# Removed st.set_page_config() as it's handled by the main app

# --- Load the trained model and scaler ---
try:
    model_path = 'parkinsons_disease_model.pkl'
    scaler_path = 'parkinsons_disease_scaler.pkl'

    if not os.path.exists(model_path):
        st.error(f"Error: Model file '{model_path}' not found. Please ensure 'parkinsons_model_training.py' was run successfully.")
        st.stop()
    if not os.path.exists(scaler_path):
        st.error(f"Error: Scaler file '{scaler_path}' not found. Please ensure 'parkinsons_model_training.py' was run successfully.")
        st.stop()

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    st.sidebar.success("Parkinson's Model and scaler loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading Parkinson's model or scaler: {e}")
    st.stop()

# --- Load Model Performance Data ---
model_performance_df = pd.DataFrame()
performance_chart_path = os.path.join('parkinsons_eda_plots', 'parkinsons_model_performance_comparison_chart.png')
try:
    performance_csv_path = 'parkinsons_model_performance.csv'
    if os.path.exists(performance_csv_path):
        model_performance_df = pd.read_csv(performance_csv_path)
        st.sidebar.success("Parkinson's Model performance data loaded successfully.")
    else:
        st.sidebar.warning("Parkinson's Model performance comparison file 'parkinsons_model_performance.csv' not found. Please run 'parkinsons_model_training.py' to generate it.")
except Exception as e:
    st.sidebar.error(f"Error loading Parkinson's model performance data: {e}")


# --- Title and Description ---
st.title("🚶‍♂️ Parkinson's Disease Prediction")
st.markdown("""
    This page allows you to predict the likelihood of Parkinson's disease based on various
    voice-related biomedical features.
""")

# --- Display Model Performance Comparison ---
if not model_performance_df.empty:
    st.subheader("📊 Model Performance Comparison")
    st.markdown("Here's how different models performed on the Parkinson's Disease dataset:")

    best_model_row = model_performance_df.loc[model_performance_df['Accuracy'].idxmax()] # Using Accuracy for best model selection
    best_model_name = best_model_row['Model']

    st.write(f"The best performing model based on Accuracy is **{best_model_name}**.")

    def highlight_best_model(s):
        is_best = s['Model'] == best_model_name
        return ['background-color: lightgreen' if is_best else '' for _ in s]

    st.dataframe(model_performance_df.style.apply(highlight_best_model, axis=1), hide_index=True)

    st.subheader("📈 Combined Model Performance Comparison Chart")
    if os.path.exists(performance_chart_path):
        st.image(performance_chart_path, caption='Comparison of Model Performance Metrics', use_column_width=True)
        st.write(f"You can find this chart saved at `{performance_chart_path}`.")
    else:
        st.warning(f"Combined model performance chart not found at `{performance_chart_path}`. Please run 'parkinsons_model_training.py' to generate it.")
else:
    st.warning("Parkinson's Model performance data is not available. Please run the training script.")

st.markdown("---")

# --- Input Features ---
st.header("📝 Enter Patient Voice Parameters for Prediction")
st.markdown("Please fill in the details below to get a prediction:")

# Define the feature columns in the exact order and capitalization expected by the model
feature_columns_order = [
    'MDVP:Fo(Hz)', 'MDVP:Fhi(Hz)', 'MDVP:Flo(Hz)', 'MDVP:Jitter(%)',
    'MDVP:Jitter(Abs)', 'MDVP:RAP', 'MDVP:PPQ', 'Jitter:DDP',
    'MDVP:Shimmer', 'MDVP:Shimmer(dB)', 'Shimmer:APQ3', 'Shimmer:APQ5',
    'MDVP:APQ', 'Shimmer:DDA', 'NHR', 'HNR', 'RPDE', 'DFA', 'spread1',
    'spread2', 'D2', 'PPE'
]

# Create input fields for each feature
input_values = {}

cols_input = st.columns(3)
with cols_input[0]:
    input_values['MDVP:Fo(Hz)'] = st.number_input('MDVP:Fo(Hz) (Avg vocal fundamental frequency)', min_value=50.0, max_value=300.0, value=150.0, step=0.1)
    input_values['MDVP:Fhi(Hz)'] = st.number_input('MDVP:Fhi(Hz) (Max vocal fundamental frequency)', min_value=50.0, max_value=600.0, value=200.0, step=0.1)
    input_values['MDVP:Flo(Hz)'] = st.number_input('MDVP:Flo(Hz) (Min vocal fundamental frequency)', min_value=50.0, max_value=250.0, value=100.0, step=0.1)
    input_values['MDVP:Jitter(%)'] = st.number_input('MDVP:Jitter(%) (Jitter in %)', min_value=0.0, max_value=0.1, value=0.005, format="%.5f")
    input_values['MDVP:Jitter(Abs)'] = st.number_input('MDVP:Jitter(Abs) (Absolute jitter)', min_value=0.0, max_value=0.001, value=0.00005, format="%.6f")
    input_values['MDVP:RAP'] = st.number_input('MDVP:RAP (Relative Amplitude Perturbation)', min_value=0.0, max_value=0.1, value=0.003, format="%.5f")
    input_values['MDVP:PPQ'] = st.number_input('MDVP:PPQ (Five-point Period Perturbation Quotient)', min_value=0.0, max_value=0.1, value=0.004, format="%.5f")
    input_values['Jitter:DDP'] = st.number_input('Jitter:DDP (DDP of Jitter)', min_value=0.0, max_value=0.3, value=0.009, format="%.5f")

with cols_input[1]:
    input_values['MDVP:Shimmer'] = st.number_input('MDVP:Shimmer (Shimmer)', min_value=0.0, max_value=0.5, value=0.03, format="%.5f")
    input_values['MDVP:Shimmer(dB)'] = st.number_input('MDVP:Shimmer(dB) (Shimmer in dB)', min_value=0.0, max_value=5.0, value=0.3, step=0.01)
    input_values['Shimmer:APQ3'] = st.number_input('Shimmer:APQ3 (Three-point Amplitude Perturbation Quotient)', min_value=0.0, max_value=0.1, value=0.015, format="%.5f")
    input_values['Shimmer:APQ5'] = st.number_input('Shimmer:APQ5 (Five-point Amplitude Perturbation Quotient)', min_value=0.0, max_value=0.1, value=0.02, format="%.5f")
    input_values['MDVP:APQ'] = st.number_input('MDVP:APQ (Period Perturbation Quotient)', min_value=0.0, max_value=0.1, value=0.025, format="%.5f")
    input_values['Shimmer:DDA'] = st.number_input('Shimmer:DDA (DDA of Shimmer)', min_value=0.0, max_value=0.3, value=0.045, format="%.5f")
    input_values['NHR'] = st.number_input('NHR (Noise-to-Harmonic Ratio)', min_value=0.0, max_value=0.5, value=0.02, format="%.5f")
    input_values['HNR'] = st.number_input('HNR (Harmonic-to-Noise Ratio)', min_value=0.0, max_value=35.0, value=20.0, step=0.1)

with cols_input[2]:
    input_values['RPDE'] = st.number_input('RPDE (Recurrence Period Density Entropy)', min_value=0.0, max_value=1.0, value=0.5, step=0.01)
    input_values['DFA'] = st.number_input('DFA (Detrended Fluctuation Analysis)', min_value=0.0, max_value=1.0, value=0.7, step=0.01)
    input_values['spread1'] = st.number_input('spread1 (Nonlinear dynamical complexity measure 1)', min_value=-10.0, max_value=0.0, value=-5.0, step=0.1)
    input_values['spread2'] = st.number_input('spread2 (Nonlinear dynamical complexity measure 2)', min_value=0.0, max_value=0.5, value=0.2, step=0.01)
    input_values['D2'] = st.number_input('D2 (Correlation Dimension)', min_value=1.0, max_value=3.0, value=2.0, step=0.01)
    input_values['PPE'] = st.number_input('PPE (Pitch Period Entropy)', min_value=0.0, max_value=0.5, value=0.2, format="%.5f")


# --- Prediction ---
st.markdown("---")
if st.button("Predict Parkinson's Disease"):
    try:
        # Create a DataFrame from input values, ensuring correct column order
        input_df = pd.DataFrame([input_values], columns=feature_columns_order)

        # Scale the input data
        scaled_input_data = scaler.transform(input_df)

        prediction_proba = model.predict_proba(scaled_input_data)[:, 1] # Probability of Parkinson's (class 1)
        prediction = model.predict(scaled_input_data)[0]

        st.subheader("Prediction Result:")
        if prediction == 1:
            st.error(f"Based on the provided information, the model predicts a **HIGH likelihood of Parkinson's Disease**.")
            st.write(f"Confidence (Probability of Disease): **{prediction_proba[0]:.2f}**")
        else:
            st.success(f"Based on the provided information, the model predicts a **LOW likelihood of Parkinson's Disease**.")
            st.write(f"Confidence (Probability of Disease): **{prediction_proba[0]:.2f}**")

        st.markdown("""
            **Disclaimer:** This prediction is based on a machine learning model and
            should not be considered a substitute for professional medical advice.
            Always consult with a qualified healthcare provider for diagnosis and treatment.
        """)
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and Scikit-learn.")