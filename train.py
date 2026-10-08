import os
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    classification_report
)
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# Create models folder if it does not exist
os.makedirs("models", exist_ok=True)

# Load new dataset
df = pd.read_csv("dataset/diabetes_prediction_dataset.csv")

print("Original dataset shape:", df.shape)

# Remove duplicate records
df = df.drop_duplicates().copy()

print("Dataset shape after removing duplicates:", df.shape)
print("\nGender counts:")
print(df["gender"].value_counts())

# Separate input features and target
X = df.drop("diabetes", axis=1)
y = df["diabetes"]

# Define columns
numeric_features = [
    "age",
    "hypertension",
    "heart_disease",
    "bmi",
    "HbA1c_level",
    "blood_glucose_level"
]

categorical_features = [
    "gender",
    "smoking_history"
]

# Preprocessing for numeric data
numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

# Combine preprocessing
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# Split dataset into training and testing data
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

# Transform training and testing data
X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# Calculate imbalance ratio for XGBoost
negative_count = (y_train == 0).sum()
positive_count = (y_train == 1).sum()
scale_pos_weight = negative_count / positive_count

# Define models
models = {
    "Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42
    ),

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        random_state=42,
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        n_jobs=-1
    )
}

results = []
trained_models = {}

# Train and evaluate models
for name, model in models.items():

    print(f"\nTraining {name}...")

    model.fit(X_train_processed, y_train)

    predictions = model.predict(X_test_processed)
    probabilities = model.predict_proba(X_test_processed)[:, 1]

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(
        y_test, predictions, zero_division=0
    )
    recall = recall_score(
        y_test, predictions, zero_division=0
    )
    f1 = f1_score(
        y_test, predictions, zero_division=0
    )
    roc_auc = roc_auc_score(y_test, probabilities)

    results.append({
        "Model": name,
        "Accuracy": accuracy,
        "Precision": precision,
        "Recall": recall,
        "F1 Score": f1,
        "ROC-AUC": roc_auc
    })

    trained_models[name] = model

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

# Compare model results
results_df = pd.DataFrame(results)
print("\nMODEL COMPARISON")
print(results_df.to_string(index=False))

# Select model with highest F1 score
best_model_name = results_df.loc[
    results_df["F1 Score"].idxmax(), "Model"
]
best_model = trained_models[best_model_name]

# Save new model and preprocessor separately
joblib.dump(best_model, "models/diabetes_model_compressed.joblib", compress=3)
joblib.dump(preprocessor, "models/diabetes_preprocessor.joblib")

print("\nBest model:", best_model_name)
print("New model saved successfully.")
print("Preprocessor saved successfully.")

print("\nClassification Report:")
best_predictions = best_model.predict(X_test_processed)
print(classification_report(y_test, best_predictions, zero_division=0))