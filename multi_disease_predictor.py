# multi_disease_predictor.py

import streamlit as st

st.set_page_config(
    page_title="Multi-Disease Prediction System",
    page_icon="🩺",
    layout="wide" # Use wide layout for better display of charts/tables
)

st.title("🩺 Welcome to the Multi-Disease Prediction System!")
st.markdown("""
    This application allows you to predict the likelihood of **Kidney Disease**,
    **Liver Disease**, and **Parkinson's Disease** using machine learning models.

    Navigate through the different prediction pages using the sidebar on the left.
    Each page provides information about the model's performance and an interactive
    form for making predictions.
""")

st.info("Please select a disease from the sidebar to start predicting!")

st.markdown("---")
st.markdown("Developed with ❤️ using Streamlit and Scikit-learn.")

# Streamlit automatically discovers and displays pages from the 'pages' directory.
# No further code is needed here to link them explicitly.