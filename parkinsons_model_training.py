import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import joblib # For saving models and scaler
import matplotlib.pyplot as plt # For plotting
import seaborn as sns # For enhanced visualizations
import os # For creating directories

print("Starting Parkinson's disease model training script with enhanced EDA and hyperparameter tuning...")

# Create a directory for plots if it doesn't exist
plots_dir = 'parkinsons_eda_plots'
os.makedirs(plots_dir, exist_ok=True)
print(f"Plots will be saved in the '{plots_dir}' directory.")

# Load the dataset
try:
    parkinsons_df = pd.read_csv('D:\Projects\Guvi\Multiple_Disease_Prediction\data\parkinsons.csv')
    print("Parkinson's disease dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'parkinsons - parkinsons.csv' not found. Please ensure the dataset is in the same directory.")
    exit()

# --- Data Preprocessing ---
print("\nPerforming Data Preprocessing...")

# Drop the 'name' column as it's an identifier and not a feature
if 'name' in parkinsons_df.columns:
    parkinsons_df = parkinsons_df.drop('name', axis=1)
    print("Dropped 'name' column.")

# Rename 'status' to 'target' for clarity
parkinsons_df = parkinsons_df.rename(columns={'status': 'target'})

# Check for missing values (Parkinson's dataset is usually clean, but good to check)
print("\nMissing values before imputation:")
print(parkinsons_df.isnull().sum())
# If there were missing values, we'd use SimpleImputer here, but typically not needed for this dataset.

# Ensure all columns are numeric (target is already 0/1)
for col in parkinsons_df.columns:
    if col != 'target': # Don't coerce target if it's already int
        parkinsons_df[col] = pd.to_numeric(parkinsons_df[col], errors='coerce')
        # If any NaNs are introduced by coerce, impute them
        if parkinsons_df[col].isnull().any():
            parkinsons_df[col] = parkinsons_df[col].fillna(parkinsons_df[col].mean())
            print(f"Imputed numerical NaNs in {col}")

print("\nData after preprocessing:")
print(parkinsons_df.head())
print(parkinsons_df.info())


# --- Enhanced Exploratory Data Analysis (EDA) ---
print("\nPerforming Enhanced EDA and saving plots...")

# 1. Target Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=parkinsons_df)
plt.title("Parkinson's Disease Target Distribution (0: Healthy, 1: Parkinson's)")
plt.xlabel("Parkinson's Status")
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['Healthy', "Parkinson's"])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(os.path.join(plots_dir, 'parkinsons_target_distribution.png'))
plt.show() # Display the plot
plt.close()
print(f"Saved Target Distribution plot to {os.path.join(plots_dir, 'parkinsons_target_distribution.png')}")

# 2. Distribution of Numerical Features
numerical_features = parkinsons_df.drop('target', axis=1).columns.tolist()
# Calculate grid size dynamically
num_features = len(numerical_features)
n_cols = 5 # Number of columns for subplots
n_rows = (num_features + n_cols - 1) // n_cols # Calculate rows needed
plt.figure(figsize=(n_cols * 4, n_rows * 3)) # Adjust figure size for many plots
for i, col in enumerate(numerical_features):
    plt.subplot(n_rows, n_cols, i + 1)
    sns.histplot(parkinsons_df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel('') # Remove x-label to prevent overlap
    plt.ylabel('') # Remove y-label to prevent overlap
plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout to make room for suptitle
plt.suptitle('Histograms and KDEs of Numerical Features', y=1.00, fontsize=16)
plt.savefig(os.path.join(plots_dir, 'parkinsons_numerical_feature_distributions.png'))
plt.show() # Display the plot
plt.close()
print(f"Saved Numerical Feature Distributions plot to {os.path.join(plots_dir, 'parkinsons_numerical_feature_distributions.png')}")

# 3. Boxplots for Numerical Features by Target
plt.figure(figsize=(n_cols * 4, n_rows * 3))
for i, col in enumerate(numerical_features):
    plt.subplot(n_rows, n_cols, i + 1)
    sns.boxplot(x='target', y=col, data=parkinsons_df)
    plt.title(f'{col} by Target')
    plt.xlabel('Parkinson\'s Status (0: No, 1: Yes)')
    plt.ylabel('') # Remove y-label to prevent overlap
plt.tight_layout(rect=[0, 0.03, 1, 0.98])
plt.suptitle('Boxplots of Numerical Features by Target Class', y=1.00, fontsize=16)
plt.savefig(os.path.join(plots_dir, 'parkinsons_numerical_features_by_target.png'))
plt.show() # Display the plot
plt.close()
print(f"Saved Boxplots of Numerical Features by Target plot to {os.path.join(plots_dir, 'parkinsons_numerical_features_by_target.png')}")

# 4. Correlation Heatmap
plt.figure(figsize=(15, 12))
sns.heatmap(parkinsons_df.corr(), annot=False, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap of Features')
plt.savefig(os.path.join(plots_dir, 'parkinsons_correlation_heatmap.png'))
plt.show() # Display the plot
plt.close()
print(f"Saved Correlation Heatmap to {os.path.join(plots_dir, 'parkinsons_correlation_heatmap.png')}")

print("EDA completed. Proceeding with model training.")

# Separate features (X) and target (y)
X = parkinsons_df.drop('target', axis=1)
y = parkinsons_df['target']
print("Features (X) and Target (y) separated.")

# Scale numerical features (all features in X are numerical)
all_features_to_scale = X.columns.tolist()

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X[all_features_to_scale])
X_scaled_df = pd.DataFrame(X_scaled, columns=all_features_to_scale)
print("Features scaled using MinMaxScaler.")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42, stratify=y)
print("Dataset split into training and testing sets.")

# --- Model Training and Hyperparameter Tuning using GridSearchCV ---
models = {
    'Logistic Regression': {
        'model': LogisticRegression(random_state=42, solver='liblinear'),
        'params': {'C': [0.01, 0.1, 1, 10, 100]}
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(random_state=42),
        'params': {'max_depth': [None, 5, 10, 20], 'min_samples_leaf': [1, 5, 10]}
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42),
        'params': {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20]}
    },
    'AdaBoost': {
        'model': AdaBoostClassifier(random_state=42),
        'params': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 1]}
    },
    'XGBoost': {
        'model': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'),
        'params': {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2], 'max_depth': [3, 5, 7]}
    }
}

performance_metrics = []
best_accuracy = 0
best_model_name = None
best_model = None

for name, config in models.items():
    print(f"\nTraining and tuning {name}...")
    grid_search = GridSearchCV(config['model'], config['params'], cv=5, scoring='accuracy', n_jobs=-1, verbose=1)
    grid_search.fit(X_train, y_train)

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

    # Track the best model based on accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = name
        best_model = tuned_model

    # Plot Confusion Matrix for the tuned model
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted 0', 'Predicted 1'],
                yticklabels=['Actual 0', 'Actual 1'])
    plt.title(f'Confusion Matrix for {name}')
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
    plt.title(f'Receiver Operating Characteristic (ROC) Curve for {name}')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(plots_dir, f'{name.lower().replace(" ", "_")}_roc_curve.png'))
    plt.show() # Display the plot
    plt.close()
    print(f"Saved ROC Curve for {name} to {os.path.join(plots_dir, f'{name.lower().replace(" ", "_")}_roc_curve.png')}")


print(f"\nBest performing model for Parkinson's Disease Prediction: {best_model_name} with Accuracy: {best_accuracy:.4f}")

# Save the best model and the scaler
joblib.dump(best_model, 'parkinsons_disease_model.pkl')
joblib.dump(scaler, 'parkinsons_disease_scaler.pkl')
print("Best Parkinson's disease model and scaler saved successfully.")

# Save model performance comparison to a CSV file
performance_df = pd.DataFrame(performance_metrics)
performance_df.to_csv('parkinsons_model_performance.csv', index=False)
print("Parkinson's model performance comparison saved to 'parkinsons_model_performance.csv'.")


# --- Generate Combined Model Performance Comparison Chart ---
print("\nGenerating Combined Model Performance Comparison Chart...")
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
performance_melted = performance_df.melt(id_vars=['Model'], value_vars=metrics_to_plot,
                                        var_name='Metric', value_name='Score')

plt.figure(figsize=(14, 8))
sns.barplot(x='Model', y='Score', hue='Metric', data=performance_melted, palette='viridis')
plt.title('Comparison of Model Performance Metrics for Parkinson\'s Prediction', fontsize=16)
plt.xlabel('Machine Learning Model', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.ylim(0, 1.0) # Metrics are between 0 and 1
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

chart_filename = 'parkinsons_model_performance_comparison_chart.png'
chart_path = os.path.join(plots_dir, chart_filename) # Save to plots directory
plt.savefig(chart_path)
plt.show() # Display the plot
plt.close()
print(f"Combined Model performance comparison chart saved to {chart_path}")

print("Parkinson's disease model training script finished.")