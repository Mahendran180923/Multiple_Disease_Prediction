import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve, confusion_matrix
import joblib # For saving models and scaler
import matplotlib.pyplot as plt # For plotting
import seaborn as sns # For enhanced visualizations
import os # For creating directories

print("Starting Kidney Disease model training script with enhanced EDA and hyperparameter tuning...")

# Create a directory for plots if it doesn't exist
plots_dir = 'kidney_eda_plots'
os.makedirs(plots_dir, exist_ok=True)
print(f"Plots will be saved in the '{plots_dir}' directory.")

# Load the dataset
try:
    kidney_df = pd.read_csv('D:\Projects\Guvi\Multiple_Disease_Prediction\data\kidney_disease.csv') # Corrected filename
    print("Kidney Disease dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'kidney_disease - kidney_disease.csv' not found. Please ensure the dataset is in the same directory.")
    exit()

# Rename columns for easier access (replace ' ' with '_', make lowercase)
kidney_df.columns = kidney_df.columns.str.lower().str.replace(' ', '_')

# Rename 'classification' to 'target' for clarity
kidney_df = kidney_df.rename(columns={'classification': 'target'})

# Handle missing values: Replace '?' with NaN and convert types where necessary
for col in ['pcv', 'wc', 'rc']:
    # Convert to string to ensure .str accessor works
    # Strip any leading/trailing whitespace
    # Replace specific problematic strings ('?', '\t?', '\t') with NaN
    kidney_df[col] = kidney_df[col].astype(str).str.strip().replace(['?', '\t?', '\t'], np.nan)
    # Convert to numeric, coercing any remaining unparseable values to NaN
    kidney_df[col] = pd.to_numeric(kidney_df[col], errors='coerce')

# --- Initial Data Exploration and Visualizations (before imputation of these numerical values) ---

# Convert 'target' to numeric for consistent plotting and handling
kidney_df['target'] = kidney_df['target'].replace({'ckd': 1, 'notckd': 0, 'yes': 1, 'no': 0})
kidney_df['target'] = pd.to_numeric(kidney_df['target'], errors='coerce')
kidney_df.dropna(subset=['target'], inplace=True) # Drop rows where target couldn't be converted
kidney_df['target'] = kidney_df['target'].astype(int)

# --- Enhanced Exploratory Data Analysis (EDA) ---
print("\nPerforming Enhanced EDA and saving plots...")

# 1. Target Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=kidney_df)
plt.title('Kidney Disease Target Distribution (0: Not CKD, 1: CKD)')
plt.xlabel('Kidney Disease Status')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['Not CKD', 'CKD'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.savefig(os.path.join(plots_dir, 'kidney_target_distribution.png'))
plt.show() # Display the plot
plt.close()
print(f"Saved Target Distribution plot to {os.path.join(plots_dir, 'kidney_target_distribution.png')}")

# Check for class imbalance
target_counts = kidney_df['target'].value_counts()
print(f"\nTarget variable distribution:\n{target_counts}")
if target_counts.min() / target_counts.max() < 0.5: # Simple heuristic for imbalance
    print("Warning: Class imbalance detected. Models will use class_weight/scale_pos_weight.")


# 2. Distribution of Numerical Features
numerical_features = kidney_df.select_dtypes(include=np.number).columns.tolist()
if 'target' in numerical_features:
    numerical_features.remove('target')
num_features = len(numerical_features)
n_cols = 5 # Number of columns for subplots
n_rows = (num_features + n_cols - 1) // n_cols # Calculate rows needed
plt.figure(figsize=(n_cols * 4, n_rows * 3))
for i, col in enumerate(numerical_features):
    plt.subplot(n_rows, n_cols, i + 1)
    sns.histplot(kidney_df[col], kde=True)
    plt.title(f'Distribution of {col}')
    plt.xlabel('')
    plt.ylabel('')
plt.tight_layout(rect=[0, 0.03, 1, 0.98])
plt.suptitle('Histograms and KDEs of Numerical Features (Kidney Disease)', y=1.00, fontsize=16)
plt.savefig(os.path.join(plots_dir, 'kidney_numerical_feature_distributions.png'))
plt.show() # Display the plot
plt.close()
print(f"Saved Numerical Feature Distributions plot to {os.path.join(plots_dir, 'kidney_numerical_feature_distributions.png')}")

# 3. Boxplots for Numerical Features by Target
plt.figure(figsize=(n_cols * 4, n_rows * 3))
for i, col in enumerate(numerical_features):
    plt.subplot(n_rows, n_cols, i + 1)
    sns.boxplot(x='target', y=col, data=kidney_df)
    plt.title(f'{col} by Target')
    plt.xlabel('Kidney Disease Status (0: No, 1: Yes)')
    plt.ylabel('')
plt.tight_layout(rect=[0, 0.03, 1, 0.98])
plt.suptitle('Boxplots of Numerical Features by Target Class (Kidney Disease)', y=1.00, fontsize=16)
plt.savefig(os.path.join(plots_dir, 'kidney_numerical_features_by_target.png'))
plt.show() # Display the plot
plt.close()
print(f"Saved Boxplots of Numerical Features by Target plot to {os.path.join(plots_dir, 'kidney_numerical_features_by_target.png')}")

# 4. Correlation Heatmap (after imputation and encoding, before scaling)
df_for_heatmap = kidney_df.copy() # Use a copy for correlation before final scaling

# Impute remaining missing values after initial conversion
# Numerical columns imputation (mean)
numerical_cols_impute = df_for_heatmap.select_dtypes(include=np.number).columns.tolist()
if 'target' in numerical_cols_impute:
    numerical_cols_impute.remove('target')

imputer_numerical = SimpleImputer(strategy='mean')
df_for_heatmap[numerical_cols_impute] = imputer_numerical.fit_transform(df_for_heatmap[numerical_cols_impute])
print("Numerical missing values imputed for heatmap generation.")

# Categorical columns imputation (mode)
categorical_cols_impute = df_for_heatmap.select_dtypes(include='object').columns.tolist()
if 'target' in categorical_cols_impute:
    categorical_cols_impute.remove('target')

imputer_categorical = SimpleImputer(strategy='most_frequent')
df_for_heatmap[categorical_cols_impute] = imputer_categorical.fit_transform(df_for_heatmap[categorical_cols_impute])
print("Categorical missing values imputed for heatmap generation.")

# Encode categorical features for heatmap
categorical_features_for_heatmap = df_for_heatmap.select_dtypes(include='object').columns.tolist()
if 'target' in categorical_features_for_heatmap:
    categorical_features_for_heatmap.remove('target')

for col in categorical_features_for_heatmap:
    le = LabelEncoder()
    df_for_heatmap[col] = le.fit_transform(df_for_heatmap[col])
    print(f"Label encoded column for heatmap: {col}")

# Drop the 'id' column if it exists in the heatmap dataframe
if 'id' in df_for_heatmap.columns:
    df_for_heatmap = df_for_heatmap.drop('id', axis=1)
    print("Dropped 'id' column for heatmap generation.")


plt.figure(figsize=(16, 12))
sns.heatmap(df_for_heatmap.corr(), annot=False, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap of Features (Kidney Disease - After Preprocessing, Before Final Scaling)')
plt.savefig(os.path.join(plots_dir, 'kidney_correlation_heatmap.png'))
plt.show() # Display the plot
plt.close()
print(f"Saved Correlation Heatmap to {os.path.join(plots_dir, 'kidney_correlation_heatmap.png')}")

print("EDA completed. Proceeding with model training.")

# --- Final Preprocessing before Model Training (re-apply on original df) ---
# Impute remaining missing values on the actual kidney_df (after initial conversions)
numerical_cols_final_impute = kidney_df.select_dtypes(include=np.number).columns.tolist()
if 'target' in numerical_cols_final_impute:
    numerical_cols_final_impute.remove('target')

imputer_numerical_final = SimpleImputer(strategy='mean')
kidney_df[numerical_cols_final_impute] = imputer_numerical_final.fit_transform(kidney_df[numerical_cols_final_impute])
print("Final numerical missing values imputed.")

categorical_cols_final_impute = kidney_df.select_dtypes(include='object').columns.tolist()
if 'target' in categorical_cols_final_impute:
    categorical_cols_final_impute.remove('target')

imputer_categorical_final = SimpleImputer(strategy='most_frequent')
kidney_df[categorical_cols_final_impute] = imputer_categorical_final.fit_transform(kidney_df[categorical_cols_final_impute])
print("Final categorical missing values imputed.")

# Encode categorical features for final training
categorical_features_final = kidney_df.select_dtypes(include='object').columns.tolist()
if 'target' in categorical_features_final:
    categorical_features_final.remove('target')

# Store LabelEncoders for each categorical column if needed for inverse transform in app
# For now, just encode
for col in categorical_features_final:
    le = LabelEncoder()
    kidney_df[col] = le.fit_transform(kidney_df[col])
    print(f"Final Label encoded column: {col}")

# Drop the 'id' column if it exists
if 'id' in kidney_df.columns:
    kidney_df = kidney_df.drop('id', axis=1)
    print("Dropped 'id' column.")


# Separate features (X) and target (y)
X = kidney_df.drop('target', axis=1)
y = kidney_df['target']
print("Features (X) and Target (y) separated.")

# Scale numerical features (all features are now numerical after encoding)
all_features_to_scale = X.columns.tolist()

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X[all_features_to_scale])
X_scaled_df = pd.DataFrame(X_scaled, columns=all_features_to_scale)
print("Features scaled using MinMaxScaler.")

# Split the dataset into training and testing sets
# Using stratify=y to maintain the proportion of target classes in splits
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42, stratify=y)
print("Dataset split into training and testing sets.")

# --- Model Training and Hyperparameter Tuning using GridSearchCV ---
# Get class weights for imbalanced datasets
class_0_count = target_counts.get(0, 0)
class_1_count = target_counts.get(1, 0)
scale_pos_weight_val = class_0_count / class_1_count if class_1_count else 1

models = {
    'Logistic Regression': {
        'model': LogisticRegression(random_state=42, solver='liblinear', class_weight='balanced'),
        'params': {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'penalty': ['l1', 'l2']}
    },
    'Decision Tree': {
        'model': DecisionTreeClassifier(random_state=42, class_weight='balanced'),
        'params': {'max_depth': [None, 5, 10, 15, 20], 'min_samples_leaf': [1, 2, 4, 8, 15]}
    },
    'Random Forest': {
        'model': RandomForestClassifier(random_state=42, class_weight='balanced'),
        'params': {'n_estimators': [50, 100, 200, 300], 'max_depth': [None, 10, 20, 30], 'min_samples_leaf': [1, 2, 4]}
    },
    'AdaBoost': {
        'model': AdaBoostClassifier(random_state=42),
        'params': {'n_estimators': [50, 100, 200, 300], 'learning_rate': [0.001, 0.01, 0.1, 1]}
    },
    'XGBoost': {
        'model': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss', scale_pos_weight=scale_pos_weight_val),
        'params': {'n_estimators': [100, 200, 300], 'learning_rate': [0.01, 0.05, 0.1, 0.2], 'max_depth': [3, 5, 7, 9], 'subsample': [0.7, 0.8, 0.9, 1.0]}
    }
}

performance_metrics = []
best_f1 = -1 # Use F1-score as primary metric for best model due to potential imbalance
best_model_name = None
best_model = None

for name, config in models.items():
    print(f"\nTraining and tuning {name}...")
    grid_search = GridSearchCV(config['model'], config['params'], cv=5, scoring='f1', n_jobs=-1, verbose=1)
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

    # Track the best model based on F1-score (more robust for imbalanced classes)
    if f1 > best_f1:
        best_f1 = f1
        best_model_name = name
        best_model = tuned_model

    # Plot Confusion Matrix for the tuned model
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False,
                xticklabels=['Predicted No CKD', 'Predicted CKD'],
                yticklabels=['Actual No CKD', 'Actual CKD'])
    plt.title(f'Confusion Matrix for {name} (Kidney Disease)')
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
    plt.title(f'Receiver Operating Characteristic (ROC) Curve for {name} (Kidney Disease)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(plots_dir, f'{name.lower().replace(" ", "_")}_roc_curve.png'))
    plt.show() # Display the plot
    plt.close()
    print(f"Saved ROC Curve for {name} to {os.path.join(plots_dir, f'{name.lower().replace(" ", "_")}_roc_curve.png')}")


print(f"\nBest performing model for Kidney Disease Prediction: {best_model_name} with F1-Score: {best_f1:.4f}")

# Save the best model and the scaler
joblib.dump(best_model, 'kidney_disease_model.pkl')
joblib.dump(scaler, 'kidney_disease_scaler.pkl')
print("Best Kidney disease model and scaler saved successfully.")

# Save model performance comparison to a CSV file
performance_df = pd.DataFrame(performance_metrics)
performance_df.to_csv('model_performance.csv', index=False) # Keep original name for consistency with app.py
print("Kidney model performance comparison saved to 'model_performance.csv'.")


# --- Generate Combined Model Performance Comparison Chart ---
print("\nGenerating Combined Model Performance Comparison Chart...")
metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'ROC AUC']
performance_melted = performance_df.melt(id_vars=['Model'], value_vars=metrics_to_plot,
                                        var_name='Metric', value_name='Score')

plt.figure(figsize=(14, 8))
sns.barplot(x='Model', y='Score', hue='Metric', data=performance_melted, palette='viridis')
plt.title('Comparison of Model Performance Metrics for Kidney Disease Prediction', fontsize=16)
plt.xlabel('Machine Learning Model', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.ylim(0, 1.0) # Metrics are between 0 and 1
plt.xticks(rotation=45, ha='right', fontsize=10)
plt.yticks(fontsize=10)
plt.legend(title='Metric', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

chart_filename = 'kidney_model_performance_comparison_chart.png'
chart_path = os.path.join(plots_dir, chart_filename) # Save to plots directory
plt.savefig(chart_path)
plt.show() # Display the plot
plt.close()
print(f"Combined Model performance comparison chart saved to {chart_path}")

print("Kidney disease model training script finished.")