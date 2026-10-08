import joblib
import numpy as np

# Load trained model
model = joblib.load("models/xgboost_model.joblib")

print("=" * 50)
print("DISEASE PREDICTION ASSISTANT")
print("=" * 50)

# Take input from user
pregnancies = float(input("Enter Pregnancies: "))
glucose = float(input("Enter Glucose: "))
blood_pressure = float(input("Enter Blood Pressure: "))
skin_thickness = float(input("Enter Skin Thickness: "))
insulin = float(input("Enter Insulin: "))
bmi = float(input("Enter BMI: "))
diabetes_pedigree = float(input("Enter Diabetes Pedigree Function: "))
age = float(input("Enter Age: "))

# Create input array
patient_data = np.array([[
    pregnancies,
    glucose,
    blood_pressure,
    skin_thickness,
    insulin,
    bmi,
    diabetes_pedigree,
    age
]])

# Make prediction
prediction = model.predict(patient_data)

print("\n" + "=" * 50)
print("PREDICTION RESULT")
print("=" * 50)

if prediction[0] == 1:
    print("Result: Diabetes Detected")
else:
    print("Result: No Diabetes Detected")