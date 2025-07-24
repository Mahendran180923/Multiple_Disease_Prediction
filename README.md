
# 🧠 Multiple Disease Prediction System

This project aims to build a scalable and accurate system for the early detection of multiple diseases, including **Kidney Disease**, **Liver Disease**, and **Parkinson's Disease**, using machine learning models in a user-friendly Streamlit web interface.

---

## 📚 Table of Contents

- [Objective](#objective)
- [System Architecture & Technologies](#system-architecture--technologies)
- [Features](#features)
- [Datasets](#datasets)
- [Exploratory Data Analysis (EDA) & Preprocessing](#exploratory-data-analysis-eda--preprocessing)
- [Model Training & Evaluation](#model-training--evaluation)
- [How to Run the Application](#how-to-run-the-application)
- [Project Structure](#project-structure)
- [Future Enhancements](#future-enhancements)
- [Contact](#contact)

---

## 🎯 Objective

- Assist in early detection of multiple diseases.
- Improve clinical decision-making with quick and accurate predictions.
- Reduce diagnostic time and cost by providing immediate insights.

---

## 🧱 System Architecture & Technologies

- **Frontend**: Streamlit UI for inputs and result display.
- **Backend**: Python ML models served via Streamlit.
- **ML Algorithms**: Logistic Regression, Decision Tree, Random Forest, AdaBoost, XGBoost.
- **Libraries**: scikit-learn, XGBoost, imbalanced-learn, pandas, numpy, matplotlib, seaborn.

---

## 🔑 Features

- **Multi-disease Prediction** (Kidney, Liver, Parkinson's)
- **User-Friendly Interface** via Streamlit
- **Interactive Visualizations** for EDA and evaluation
- **Scalable and Modular Design**
- **Secure Data Handling** (focuses on local privacy, future-ready for secure deployment)

---

## 📊 Datasets

- `kidney_disease.csv`: CKD parameters
- `indian_liver_patient.csv`: Liver enzyme levels and metadata
- `parkinsons.csv`: Voice metrics related to Parkinson's

---

## 📈 Exploratory Data Analysis (EDA) & Preprocessing

- Missing value imputation
- Label encoding of categorical variables
- MinMaxScaler for feature normalization
- Correlation heatmaps, KDE plots, boxplots, and target distribution analysis
- Saved as `.png` in respective `*_eda_plots/` directories

---

## 🤖 Model Training & Evaluation

- Models trained:
  - Logistic Regression
  - Decision Tree
  - Random Forest
  - AdaBoost
  - XGBoost
- Hyperparameter tuning via `GridSearchCV`
- Imbalance handling with `SMOTE` (for Liver Disease)
- Metrics:
  - Accuracy, Precision, Recall, F1-score, ROC AUC
- Visual evaluation with Confusion Matrix, ROC Curve, Metric Comparison Bar Chart

---

## 🧪 How to Run the Application

### 🔧 Prerequisites
- Python 3.8+
- pip

### 📥 Installation

```bash
git clone https://github.com/Mahendran180923/Multiple_Disease_Prediction.git
cd Multiple_Disease_Prediction
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

### 🧠 Run Training Scripts

```bash
python model_training.py                 # Kidney
python liver_model_training.py          # Liver
python parkinsons_model_training.py     # Parkinson's
```

### 🌐 Launch Apps

```bash
streamlit run app.py             # Kidney
streamlit run liver_app.py       # Liver
streamlit run parkinsons_app.py  # Parkinson's
```

---

## 📁 Project Structure

```
Multiple_Disease_Prediction/
├── *.csv                                # Datasets
├── app.py                               # Kidney app
├── liver_app.py                         # Liver app
├── parkinsons_app.py                    # Parkinson's app
├── *_model_training.py                  # Model training scripts
├── *_model.pkl, *_scaler.pkl            # Trained models and scalers
├── *_eda_plots/                         # Saved visualizations
├── *_model_performance.csv              # Model evaluation
└── README.md                            # This file
```

---

## 🔮 Future Enhancements

- Predict more diseases
- User authentication
- Cloud deployment (AWS, GCP, Azure)
- Model explainability (e.g., SHAP, LIME)
- Real-time data integration

---

## 📬 Contact

**Mahendran**  
GitHub: [Mahendran180923](https://github.com/Mahendran180923)
