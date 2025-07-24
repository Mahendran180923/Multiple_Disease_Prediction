import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import joblib # For saving models and scaler
import matplotlib.pyplot as plt # For plotting
import seaborn as sns # For enhanced visualizations
import os # For creating directories
from imblearn.over_sampling import SMOTE # For handling imbalanced data

print("Starting Liver Disease model training script with enhanced preprocessing, EDA, hyperparameter tuning, and SMOTE for imbalance handling...")

# Create a directory for plots if it doesn't exist
plots_dir = 'liver_eda_plots'
os.makedirs(plots_dir, exist_ok=True)
print(f"Plots will be saved in the '{plots_dir}' directory.")

# Load the dataset
try:
    liver_df = pd.read_csv('indian_liver_patient.csv')
    print("Indian Liver Patient dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'indian_liver_patient - indian_liver_patient.csv' not found. Please ensure the dataset is in the same directory.")
    exit()

# --- Data Preprocessing ---
print("\nPerforming Data Preprocessing...")

# Handle missing values: Albumin_and_Globulin_Ratio has some NaNs
if liver_df['Albumin_and_Globulin_Ratio'].isnull().sum() > 0:
    liver_df['Albumin_and_Globulin_Ratio'] = liver_df['Albumin_and_Globulin_Ratio'].fillna(liver_df['Albumin_and_Globulin_Ratio'].mean())
    print("Imputed missing values in 'Albumin_and_Globulin_Ratio' with its mean.")

# Encode 'Gender' column (categorical feature)
gender_encoder = LabelEncoder()
liver_df['Gender'] = gender_encoder.fit_transform(liver_df['Gender'])
print("Encoded 'Gender' column.")

# Rename 'Dataset' to 'target' and remap to 0 and 1
# Original: 1 (disease), 2 (no disease) -> New: 1 (disease), 0 (no disease)
liver_df = liver_df.rename(columns={'Dataset': 'target'})
liver_df['target'] = liver_df['target'].map({1: 1, 2: 0})
print("Remapped 'target' column (1: Disease, 0: No Disease).")

print("\nData after preprocessing:")
print(liver_df.head())
print(liver_df.info())


# --- Enhanced Exploratory Data Analysis (EDA) ---
print("\nPerforming Enhanced EDA and saving plots...")

# 1. Target Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=liver_df)
plt.title("Liver Disease Target Distribution (0: No Disease, 1: Disease)")
plt.xlabel("Liver Disease Status")
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['No Disease', 'Disease'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(os.path.join(plots_dir, 'liver_target_distribution.png'))
plt.show() # Display the plot
plt.close()
print(f"Saved Target Distribution plot to {os.path.join(plots_dir, 'liver_target_distribution.png')}")

# Check for class imbalance
target_counts = liver_df['target'].value_counts()
print(f"\nTarget variable distribution before SMOTE:\n{target_counts}")
if target_counts.min() / target_counts.max() < 0.5: # Simple heuristic for imbalance
    print("Class imbalance detected. Applying SMOTE to training data.")

# 2. Distribution of Numerical Features
numerical_features = liver_df.drop('target', axis=1).columns.tolist()
num_features = len(numerical_features)
n_cols = 4
n_rows = (num_features + n_cols - 1) // n_cols
plt.figure(figsize=(n_cols * 4, n_rows * 3))
for i, col in enumerate(numerical_features):
    plt.subplot(n_rows, n_cols, i + 1)
    sns.histplot(liver_df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel('')
    plt.ylabel('')
plt.tight_layout(rect=[0, 0.03, 1, 0.98])
plt.suptitle('Histograms and KDEs of Numerical Features (Liver Disease)', y=1.00, fontsize=16)
plt.savefig(os.path.join(plots_dir, 'liver_numerical_feature_distributions.png'))
plt.show() # Display the plot
plt.close()
print(f"Saved Numerical Feature Distributions plot to {os.path.join(plots_dir, 'liver_numerical_feature_distributions.png')}")

# 3. Boxplots for Numerical Features by Target
plt.figure(figsize=(n_cols * 4, n_rows * 3))
for i, col in enumerate(numerical_features):
    plt.subplot(n_rows, n_cols, i + 1)
    sns.boxplot(x='target', y=col, data=liver_df)
    plt.title(f'{col} by Target')
    plt.xlabel('Liver Disease Status (0: No, 1: Yes)')
    plt.ylabel('')
plt.tight_layout(rect=[0, 0.03, 1, 0.98])
plt.suptitle('Boxplots of Numerical Features by Target Class (Liver Disease)', y=1.00, fontsize=16)
plt.savefig(os.path.join(plots_dir, 'liver_numerical_features_by_target.png'))
plt.show() # Display the plot
plt.close()
print(f"Saved Boxplots of Numerical Features by Target plot to {os.path.join(plots_dir, 'liver_numerical_features_by_target.png')}")

# 4. Correlation Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(liver_df.corr(), annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap of Features (Liver Disease)')
plt.savefig(os.path.join(plots_dir, 'liver_correlation_heatmap.png'))
plt.show() # Display the plot
plt.close()
print(f"Saved Correlation Heatmap to {os.path.join(plots_dir, 'liver_correlation_heatmap.png')}")

print("EDA completed. Proceeding with model training.")

# Separate features (X) and target (y)
X = liver_df.drop('target', axis=1)
y = liver_df['target']
print("Features (X) and Target (y) separated.")

# Scale numerical features
feature_names = X.columns.tolist()
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)
X_scaled_df = pd.DataFrame(X_scaled, columns=feature_names)
print("Features scaled using MinMaxScaler.")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42, stratify=y)
print("Dataset split into training and testing sets.")

# Apply SMOTE to the training data
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
print(f"Training data resampled using SMOTE. Original shape: {X_train.shape}, Resampled shape: {X_train_resampled.shape}")
print(f"Target variable distribution after SMOTE on training data:\n{pd.Series(y_train_resampled).value_counts()}")


# --- Model Training and Hyperparameter Tuning using GridSearchCV ---
models = {
    'Logistic Regression': {
        'model': LogisticRegression(random_state=42, solver='liblinear'), # Removed class_weight since SMOTE handles imbalance
        'params': {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(random_state=42), # Removed class_weight
        'params': {'max_depth': [None, 5, 10, 15, 20], 'min_samples_leaf': [1, 2, 4, 8, 15]}
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42), # Removed class_weight
        'params': {'n_estimators': [50, 100, 200, 300], 'max_depth': [None, 10, 20, 30], 'min_samples_leaf': [1, 2, 4]}
    },
    'AdaBoost': {
        'model': AdaBoostClassifier(random_state=42),
        'params': {'n_estimators': [50, 100, 200, 300], 'learning_rate': [0.001, 0.01, 0.1, 1]}
    },
    'XGBoost': {
        'model': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'), # Removed scale_pos_weight
        'params': {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1, 0.2], 'max_depth': [3, 5, 7, 9], 'subsample': [0.7, 0.8, 0.9, 1.0]}
    }
}

performance_metrics = []
best_f1 = -1 # Use F1-score as primary metric for best model due to potential imbalance
best_model_name = None
best_model = None

for name, config in models.items():
    print(f"\nTraining and tuning {name}...")
    # Use resampled data for training
    grid_search = GridSearchCV(config['model'], config['params'], cv=5, scoring='f1', n_jobs=-1, verbose=1)
    grid_search.fit(X_train_resampled, y_train_resampled)

    tuned_model = grid_search.best_estimator_
    y_pred = tuned_model.predict(X_test)
    y_pred_proba = tuned_model.predict_proba(X_test)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, zero_division=0)
    recall = recall_score(y_test, y_pred, zero_division=0)
    f1 = f1_score(y_test, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test, y_pred_proba)

    print(f"{name} Performance (Tuned):")
    print(f"  Best Parameters: {grid_search.best_params_}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  ROC AUC: {roc_auc:.4f}")

    performance_metrics.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1,
        'ROC AUC': roc_auc,
        'Best Params': grid_search.best_params_
    })

    # Track the best model based on F1-score (more robust for imbalanced classes)
    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name
        best_model = tuned_model

    # Plot Confusion Matrix for the tuned model
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted No Disease', 'Predicted Disease'],
                yticklabels=['Actual No Disease', 'Actual Disease'])
    plt.title(f'Confusion Matrix for {name} (Liver Disease)')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.savefig(os.path.join(plots_dir, f'{name.lower().replace(" ", "_")}_confusion_matrix.png'))
    plt.show() # Display the plot
    plt.close()
    print(f"Saved Confusion Matrix for {name} to {os.path.join(plots_dir, f'{name.lower().replace(" ", "_")}_confusion_matrix.png')}")

    # Plot ROC Curve for the tuned model
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    plt.figure(figsize=(6, 5))
    plt.plot(fpr, tpr, color='orange', label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'Receiver Operating Characteristic (ROC) Curve for {name} (Liver Disease)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(plots_dir, f'{name.lower().replace(" ", "_")}_roc_curve.png'))
    plt.show() # Display the plot
    plt.close()
    print(f"Saved ROC Curve for {name} to {os.path.join(plots_dir, f'{name.lower().replace(" ", "_")}_roc_curve.png')}")


print(f"\nBest performing model for Liver Disease Prediction: {best_model_name} with F1-Score: {best_f1:.4f}")

# Save the best model, scaler, and gender encoder
joblib.dump(best_model, 'liver_disease_model.pkl')
joblib.dump(scaler, 'liver_disease_scaler.pkl')
joblib.dump(gender_encoder, 'liver_gender_encoder.pkl')
print("Best Liver disease model, scaler, and gender encoder saved successfully.")

# Save model performance comparison to a CSV file
performance_df = pd.DataFrame(performance_metrics)
performance_df.to_csv('liver_model_performance.csv', index=False)
print("Liver model performance comparison saved to 'liver_model_performance.csv'.")


# --- Generate Combined Model Performance Comparison Chart ---
print("\nGenerating Combined Model Performance Comparison Chart...")
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
performance_melted = performance_df.melt(id_vars=['Model'], value_vars=metrics_to_plot,
                                        var_name='Metric', value_name='Score')

plt.figure(figsize=(14, 8))
sns.barplot(x='Model', y='Score', hue='Metric', data=performance_melted, palette='viridis')
plt.title('Comparison of Model Performance Metrics for Liver Disease Prediction', fontsize=16)
plt.xlabel('Machine Learning Model', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.ylim(0, 1.0) # Metrics are between 0 and 1
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

chart_filename = 'liver_model_performance_comparison_chart.png'
chart_path = os.path.join(plots_dir, chart_filename) # Save to plots directory
plt.savefig(chart_path)
plt.show() # Display the plot
plt.close()
print(f"Combined Model performance comparison chart saved to {chart_path}")

print("Liver disease model training script finished.")