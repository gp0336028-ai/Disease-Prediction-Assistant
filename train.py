import os
import joblib
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from xgboost import XGBClassifier

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay


# ============================================
# CREATE MODELS FOLDER
# ============================================

os.makedirs("models", exist_ok=True)


# ============================================
# LOAD DATASET
# ============================================

data = pd.read_csv("dataset/diabetes.csv")

print("Dataset loaded successfully!")
print("Shape:", data.shape)


# ============================================
# SEPARATE FEATURES AND TARGET
# ============================================

X = data.drop("Outcome", axis=1)
y = data["Outcome"]

print("\nFeatures shape:", X.shape)
print("Target shape:", y.shape)


# ============================================
# REPLACE INVALID ZERO VALUES
# ============================================

columns_with_zero_as_missing = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

X[columns_with_zero_as_missing] = X[
    columns_with_zero_as_missing
].replace(0, np.nan)


# ============================================
# TRAIN-TEST SPLIT
# ============================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain-Test Split:")
print("X_train:", X_train.shape)
print("X_test :", X_test.shape)
print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# ============================================
# PREPROCESSING PIPELINE
# ============================================

preprocessor = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]
)


# ============================================
# FIT PREPROCESSOR
# ============================================

X_train_processed = preprocessor.fit_transform(X_train)

X_test_processed = preprocessor.transform(X_test)


# ============================================
# SAVE PREPROCESSOR
# ============================================

joblib.dump(
    preprocessor,
    "models/preprocessor.joblib"
)

print("\nPreprocessor saved successfully!")


print("\nPreprocessing completed successfully!")
print(
    "Processed training data shape:",
    X_train_processed.shape
)
print(
    "Processed testing data shape:",
    X_test_processed.shape
)


# ============================================
# LOGISTIC REGRESSION
# ============================================

logistic_model = LogisticRegression(
    max_iter=1000,
    random_state=42
)

logistic_model.fit(
    X_train_processed,
    y_train
)

y_pred = logistic_model.predict(
    X_test_processed
)

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\n" + "=" * 50)
print("LOGISTIC REGRESSION RESULTS")
print("=" * 50)

print(f"Accuracy: {accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred
    )
)


# ============================================
# RANDOM FOREST
# ============================================

random_forest_model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

random_forest_model.fit(
    X_train_processed,
    y_train
)

rf_pred = random_forest_model.predict(
    X_test_processed
)

rf_accuracy = accuracy_score(
    y_test,
    rf_pred
)

print("\n" + "=" * 50)
print("RANDOM FOREST RESULTS")
print("=" * 50)

print(f"Accuracy: {rf_accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        rf_pred
    )
)


# ============================================
# XGBOOST
# ============================================

xgb_model = XGBClassifier(
    n_estimators=200,
    learning_rate=0.05,
    max_depth=4,
    random_state=42,
    eval_metric="logloss"
)

xgb_model.fit(
    X_train_processed,
    y_train
)

xgb_pred = xgb_model.predict(
    X_test_processed
)

xgb_accuracy = accuracy_score(
    y_test,
    xgb_pred
)

print("\n" + "=" * 50)
print("XGBOOST RESULTS")
print("=" * 50)

print(f"Accuracy: {xgb_accuracy * 100:.2f}%")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        xgb_pred
    )
)


# ============================================
# SAVE XGBOOST MODEL
# ============================================

joblib.dump(
    xgb_model,
    "models/xgboost_model.joblib"
)

print("\n" + "=" * 50)
print("MODEL SAVED SUCCESSFULLY")
print("=" * 50)

print(
    "Saved at: models/xgboost_model.joblib"
)


# ============================================
# MODEL COMPARISON
# ============================================

print("\n" + "=" * 50)
print("MODEL COMPARISON")
print("=" * 50)

print(
    f"Logistic Regression : "
    f"{accuracy * 100:.2f}%"
)

print(
    f"Random Forest       : "
    f"{rf_accuracy * 100:.2f}%"
)

print(
    f"XGBoost             : "
    f"{xgb_accuracy * 100:.2f}%"
)


models_accuracy = {
    "Logistic Regression": accuracy,
    "Random Forest": rf_accuracy,
    "XGBoost": xgb_accuracy
}

best_model_name = max(
    models_accuracy,
    key=models_accuracy.get
)

print(
    f"\nBest Model: {best_model_name}"
)

print(
    f"Best Accuracy: "
    f"{models_accuracy[best_model_name] * 100:.2f}%"
)


# ============================================
# XGBOOST CONFUSION MATRIX
# ============================================

cm = confusion_matrix(
    y_test,
    xgb_pred
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "No Diabetes",
        "Diabetes"
    ]
)

display.plot()

plt.title(
    "XGBoost Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    "models/xgboost_confusion_matrix.png"
)

plt.close()


# ============================================
# XGBOOST FEATURE IMPORTANCE
# ============================================

feature_names = X.columns

importance = (
    xgb_model.feature_importances_
)

plt.figure(figsize=(10, 6))

plt.barh(
    feature_names,
    importance
)

plt.xlabel("Importance")
plt.ylabel("Features")

plt.title(
    "XGBoost Feature Importance"
)

plt.tight_layout()

plt.savefig(
    "models/xgboost_feature_importance.png"
)

plt.close()


print(
    "\nEvaluation charts saved successfully!"
)

print(
    "\nALL TRAINING STEPS COMPLETED SUCCESSFULLY!"
)