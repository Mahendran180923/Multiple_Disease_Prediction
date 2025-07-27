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
    model_path = 'kidney_disease_model.pkl'
    scaler_path = 'kidney_disease_scaler.pkl'
    
    # Check if files exist relative to the root where streamlit is run
    if not os.path.exists(model_path):
        st.error(f"Error: Model file '{model_path}' not found. Please ensure 'model_training.py' was run successfully.")
        st.stop()
    if not os.path.exists(scaler_path):
        st.error(f"Error: Scaler file '{scaler_path}' not found. Please ensure 'model_training.py' was run successfully.")
        st.stop()

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    st.sidebar.success("Kidney Model and scaler loaded successfully!")
except Exception as e:
    st.sidebar.error(f"Error loading Kidney model or scaler: {e}")
    st.stop()

# --- Load Model Performance Data ---
model_performance_df = pd.DataFrame()
performance_chart_path = os.path.join('kidney_eda_plots', 'kidney_model_performance_comparison_chart.png')
try:
    performance_csv_path = 'model_performance.csv' # This is the kidney performance CSV
    if os.path.exists(performance_csv_path):
        model_performance_df = pd.read_csv(performance_csv_path)
        st.sidebar.success("Kidney Model performance data loaded successfully.")
    else:
        st.sidebar.warning("Kidney Model performance comparison file 'model_performance.csv' not found. Please run 'model_training.py' to generate it.")
except Exception as e:
    st.sidebar.error(f"Error loading Kidney model performance data: {e}")


# --- Title and Description ---
st.title("🫘 Kidney Disease Prediction")
st.markdown("""
    This page allows you to predict the likelihood of Kidney Disease based on various patient parameters.
""")

# --- Display Model Performance Comparison ---
if not model_performance_df.empty:
    st.subheader("📊 Model Performance Comparison")
    st.markdown("Here's how different models performed on the Kidney Disease dataset:")

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
        st.warning(f"Combined model performance chart not found at `{performance_chart_path}`. Please run 'model_training.py' to generate it.")
else:
    st.warning("Kidney Model performance data is not available. Please run the training script.")

st.markdown("---")

# --- Input Features ---
st.header("📝 Enter Patient Details for Prediction")
st.markdown("Please fill in the details below to get a prediction:")

# Define the exact order of features as seen by the model during training
feature_columns_order = [
    'age', 'bp', 'sg', 'al', 'su',
    'rbc', 'pc', 'pcc', 'ba',
    'bgr', 'bu', 'sc', 'sod', 'pot',
    'hemo', 'pcv', 'wc', 'rc',
    'htn', 'dm', 'cad', 'appet',
    'pe', 'ane'
]

# Mappings for categorical features (consistent with model_training.py)
rbc_options = {'normal': 1, 'abnormal': 0}
pc_options = {'normal': 1, 'abnormal': 0}
pcc_options = {'notpresent': 1, 'present': 0}
ba_options = {'notpresent': 1, 'present': 0}
htn_options = {'no': 0, 'yes': 1}
dm_options = {'no': 0, 'yes': 1}
cad_options = {'no': 0, 'yes': 1}
appet_options = {'good': 0, 'poor': 1}
pe_options = {'no': 0, 'yes': 1}
ane_options = {'no': 0, 'yes': 1}

# Create input fields (organized in columns)
cols_input = st.columns(3)

with cols_input[0]:
    age = st.number_input('Age', min_value=1, max_value=100, value=45)
    blood_pressure = st.number_input('Blood Pressure (mmHg)', min_value=0, max_value=200, value=80)
    specific_gravity = st.selectbox('Specific Gravity', options=[1.005, 1.010, 1.015, 1.020, 1.025])
    albumin = st.number_input('Albumin (0-5)', min_value=0, max_value=5, value=0)
    sugar = st.number_input('Sugar (0-5)', min_value=0, max_value=5, value=0)

with cols_input[1]:
    red_blood_cells_input = st.selectbox('Red Blood Cells', options=list(rbc_options.keys()))
    pus_cell_input = st.selectbox('Pus Cell', options=list(pc_options.keys()))
    pus_cell_clumps_input = st.selectbox('Pus Cell Clumps', options=list(pcc_options.keys()))
    bacteria_input = st.selectbox('Bacteria', options=list(ba_options.keys()))
    blood_glucose_random = st.number_input('Blood Glucose Random (mg/dL)', min_value=0, max_value=500, value=120)
    blood_urea = st.number_input('Blood Urea (mg/dL)', min_value=0, max_value=300, value=40)

with cols_input[2]:
    serum_creatinine = st.number_input('Serum Creatinine (mg/dL)', min_value=0.0, max_value=20.0, value=1.0, step=0.1)
    sodium = st.number_input('Sodium (mEq/L)', min_value=0, max_value=160, value=135)
    potassium = st.number_input('Potassium (mEq/L)', min_value=0.0, max_value=10.0, value=4.0, step=0.1)
    hemoglobin = st.number_input('Hemoglobin (g/dL)', min_value=0.0, max_value=20.0, value=13.0, step=0.1)
    packed_cell_volume = st.number_input('Packed Cell Volume', min_value=0, max_value=60, value=40)
    white_blood_cell_count = st.number_input('White Blood Cell Count (cells/cmm)', min_value=0, max_value=20000, value=7000, step=100)
    red_blood_cell_count = st.number_input('Red Blood Cell Count (millions/cmm)', min_value=0.0, max_value=10.0, value=5.0, step=0.1)

cols_input_2 = st.columns(3)
with cols_input_2[0]:
    hypertension_input = st.selectbox('Hypertension', options=list(htn_options.keys()))
    diabetes_mellitus_input = st.selectbox('Diabetes Mellitus', options=list(dm_options.keys()))
with cols_input_2[1]:
    coronary_artery_disease_input = st.selectbox('Coronary Artery Disease', options=list(cad_options.keys()))
    appetite_input = st.selectbox('Appetite', options=list(appet_options.keys()))
with cols_input_2[2]:
    pedal_edema_input = st.selectbox('Pedal Edema', options=list(pe_options.keys()))
    anemia_input = st.selectbox('Anemia', options=list(ane_options.keys()))


# --- Prediction Button ---
st.markdown("---")
if st.button("Predict Kidney Disease"):
    try:
        # Map categorical inputs to numerical values
        red_blood_cells = rbc_options[red_blood_cells_input]
        pus_cell = pc_options[pus_cell_input]
        pus_cell_clumps = pcc_options[pus_cell_clumps_input]
        bacteria = ba_options[bacteria_input]
        hypertension = htn_options[hypertension_input]
        diabetes_mellitus = dm_options[diabetes_mellitus_input]
        coronary_artery_disease = cad_options[coronary_artery_disease_input]
        appetite = appet_options[appetite_input]
        pedal_edema = pe_options[pedal_edema_input]
        anemia = ane_options[anemia_input]

        # Create a dictionary of all inputs using the *abbreviated* names as keys
        input_data = {
            'age': age,
            'bp': blood_pressure,
            'sg': specific_gravity,
            'al': albumin,
            'su': sugar,
            'rbc': red_blood_cells,
            'pc': pus_cell,
            'pcc': pus_cell_clumps,
            'ba': bacteria,
            'bgr': blood_glucose_random,
            'bu': blood_urea,
            'sc': serum_creatinine,
            'sod': sodium,
            'pot': potassium,
            'hemo': hemoglobin,
            'pcv': packed_cell_volume,
            'wc': white_blood_cell_count,
            'rc': red_blood_cell_count,
            'htn': hypertension,
            'dm': diabetes_mellitus,
            'cad': coronary_artery_disease,
            'appet': appetite,
            'pe': pedal_edema,
            'ane': anemia
        }

        # Create a DataFrame with the input data, ensuring the correct column order
        input_df = pd.DataFrame([input_data], columns=feature_columns_order)

        # Scale the input data
        scaled_input_data = scaler.transform(input_df)

        prediction_proba = model.predict_proba(scaled_input_data)[:, 1] # Probability of CKD (class 1)
        prediction = model.predict(scaled_input_data)[0]

        st.subheader("Prediction Result:")
        if prediction == 1:
            st.error(f"Based on the provided information, the model predicts a **HIGH likelihood of Kidney Disease**.")
            st.write(f"Confidence (Probability of CKD): **{prediction_proba[0]:.2f}**")
        else:
            st.success(f"Based on the provided information, the model predicts a **LOW likelihood of Kidney Disease**.")
            st.write(f"Confidence (Probability of CKD): **{(1 - prediction_proba[0]):.2f}**") # Probability of not CKD

        st.markdown("""
            **Disclaimer:** This prediction is based on a machine learning model and
            should not be considered a substitute for professional medical advice.
            Always consult with a qualified healthcare provider for diagnosis and treatment.
        """)
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

st.markdown("---")
st.markdown("Built with ❤️ using Streamlit and Scikit-learn.")