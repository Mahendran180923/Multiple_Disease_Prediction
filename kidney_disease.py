import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import joblib # For saving models and scaler
import matplotlib.pyplot as plt # For plotting
import seaborn as sns # For enhanced visualizations

print("Starting model training script...")

# Load the dataset
try:
    kidney_df = pd.read_csv('kidney_disease.csv')
    print("Dataset loaded successfully.")
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

# 1. Target Distribution
plt.figure(figsize=(6, 4))
sns.countplot(x='target', data=kidney_df)
plt.title('Target Distribution (0: Not CKD, 1: CKD)')
plt.xlabel('Kidney Disease Status')
plt.ylabel('Count')
plt.xticks(ticks=[0, 1], labels=['Not CKD', 'CKD'])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show() # Display the plot
plt.close() # Close the figure

# 2. Boxplots for original numerical features (before imputation of remaining NaNs)
original_numerical_cols = kidney_df.select_dtypes(include=np.number).columns.tolist()
if 'target' in original_numerical_cols:
    original_numerical_cols.remove('target')

plt.figure(figsize=(18, 12))
for i, col in enumerate(original_numerical_cols):
    plt.subplot(3, 5, i + 1) # Adjust subplot grid based on number of numerical features
    sns.boxplot(y=kidney_df[col])
    plt.title(f'Boxplot of {col}')
    plt.ylabel('') # Hide y-label for cleaner look
plt.tight_layout()
plt.suptitle('Boxplots of Numerical Features (Before Final Imputation)', y=1.02, fontsize=16)
plt.show() # Display the plot
plt.close() # Close the figure


# Impute remaining missing values after initial conversion
# Numerical columns imputation (mean)
numerical_cols_impute = kidney_df.select_dtypes(include=np.number).columns.tolist()
if 'target' in numerical_cols_impute:
    numerical_cols_impute.remove('target')

imputer_numerical = SimpleImputer(strategy='mean')
kidney_df[numerical_cols_impute] = imputer_numerical.fit_transform(kidney_df[numerical_cols_impute])
print("Numerical missing values imputed.")

# Categorical columns imputation (mode)
categorical_cols_impute = kidney_df.select_dtypes(include='object').columns.tolist()
if 'target' in categorical_cols_impute:
    categorical_cols_impute.remove('target')

imputer_categorical = SimpleImputer(strategy='most_frequent')
kidney_df[categorical_cols_impute] = imputer_categorical.fit_transform(kidney_df[categorical_cols_impute])
print("Categorical missing values imputed.")


# Encode categorical features
categorical_features = kidney_df.select_dtypes(include='object').columns.tolist()
if 'target' in categorical_features:
    categorical_features.remove('target')

for col in categorical_features:
    le = LabelEncoder()
    kidney_df[col] = le.fit_transform(kidney_df[col])
    print(f"Label encoded column: {col}")

# Drop the 'id' column if it exists
if 'id' in kidney_df.columns:
    kidney_df = kidney_df.drop('id', axis=1)
    print("Dropped 'id' column.")

# --- Correlation Heatmap (after imputation and encoding, before scaling) ---
# Create a copy to not modify the original kidney_df for scaling
df_for_heatmap = kidney_df.copy()

plt.figure(figsize=(16, 12))
sns.heatmap(df_for_heatmap.corr(), annot=False, cmap='coolwarm', fmt=".2f", linewidths=.5)
plt.title('Correlation Heatmap of Features (After Preprocessing, Before Scaling)')
plt.show() # Display the plot
plt.close() # Close the figure


# Separate features (X) and target (y)
X = kidney_df.drop('target', axis=1)
y = kidney_df['target']
print("Features (X) and Target (y) separated.")

# Scale numerical features (all features are now numerical after encoding)
# Identify all columns in X for scaling.
all_features_to_scale = X.columns.tolist()

scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X[all_features_to_scale])
X_scaled_df = pd.DataFrame(X_scaled, columns=all_features_to_scale) # Convert back to DataFrame
print("Features scaled using MinMaxScaler.")

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled_df, y, test_size=0.2, random_state=42)
print("Dataset split into training and testing sets.")

# --- Model Training and Evaluation ---
models = {
    'Logistic Regression': LogisticRegression(random_state=42),
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'AdaBoost': AdaBoostClassifier(random_state=42),
    'XGBoost': XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
}

performance_metrics = []
best_accuracy = 0
best_model_name = None
best_model = None

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    print(f"{name} Performance:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")

    performance_metrics.append({
        'Model': name,
        'Accuracy': accuracy,
        'Precision': precision,
        'Recall': recall,
        'F1-Score': f1
    })

    # Track the best model based on accuracy
    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_model_name = name
        best_model = model

print(f"\nBest performing model: {best_model_name} with Accuracy: {best_accuracy:.4f}")

# Save the best model and the scaler
joblib.dump(best_model, 'kidney_disease_model.pkl')
joblib.dump(scaler, 'kidney_disease_scaler.pkl')
print("Best model and scaler saved successfully.")

# Save model performance comparison to a CSV file
performance_df = pd.DataFrame(performance_metrics)
performance_df.to_csv('model_performance.csv', index=False)
print("Model performance comparison saved to 'model_performance.csv'.")

print("Model training script finished.")