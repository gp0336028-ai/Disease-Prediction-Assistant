import streamlit as st
import joblib
import pandas as pd


# ============================================
# LOAD MODEL & PREPROCESSOR
# ============================================

model = joblib.load("models/xgboost_model.joblib")
preprocessor = joblib.load("models/preprocessor.joblib")


# ============================================
# PAGE CONFIGURATION
# ============================================

st.set_page_config(
    page_title="Disease Prediction Assistant",
    page_icon="🩺",
    layout="centered"
)


# ============================================
# CUSTOM CSS
# ============================================

st.markdown("""
<style>

.main-title {
    text-align: center;
    font-size: 42px;
    font-weight: bold;
    margin-bottom: 5px;
}

.subtitle {
    text-align: center;
    font-size: 17px;
    margin-bottom: 25px;
}

.result-box {
    padding: 20px;
    border-radius: 12px;
    margin-top: 15px;
}

</style>
""", unsafe_allow_html=True)


# ============================================
# HEADER
# ============================================

st.markdown(
    '<div class="main-title">🩺 Disease Prediction Assistant</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">'
    'Enter patient health information to estimate diabetes risk.'
    '</div>',
    unsafe_allow_html=True
)

st.divider()


# ============================================
# PATIENT INFORMATION
# ============================================

st.subheader("👤 Patient Information")

col1, col2 = st.columns(2)

with col1:

    pregnancies = st.number_input(
        "Pregnancies",
        min_value=0,
        max_value=20,
        value=1
    )

    glucose = st.number_input(
        "Glucose",
        min_value=0.0,
        max_value=300.0,
        value=120.0
    )

    blood_pressure = st.number_input(
        "Blood Pressure",
        min_value=0.0,
        max_value=200.0,
        value=70.0
    )

    skin_thickness = st.number_input(
        "Skin Thickness",
        min_value=0.0,
        max_value=100.0,
        value=20.0
    )


with col2:

    insulin = st.number_input(
        "Insulin",
        min_value=0.0,
        max_value=900.0,
        value=80.0
    )

    bmi = st.number_input(
        "BMI",
        min_value=0.0,
        max_value=70.0,
        value=25.0
    )

    diabetes_pedigree = st.number_input(
        "Diabetes Pedigree Function",
        min_value=0.0,
        max_value=3.0,
        value=0.5
    )

    age = st.number_input(
        "Age",
        min_value=1,
        max_value=120,
        value=30
    )


st.divider()


# ============================================
# PREDICTION BUTTON
# ============================================

if st.button(
    "🔍 Predict Diabetes Risk",
    use_container_width=True
):

    # Create input DataFrame
    patient_data = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose,
        "BloodPressure": blood_pressure,
        "SkinThickness": skin_thickness,
        "Insulin": insulin,
        "BMI": bmi,
        "DiabetesPedigreeFunction": diabetes_pedigree,
        "Age": age
    }])


    # ========================================
    # HANDLE ZERO VALUES
    # ========================================

    columns_with_zero_as_missing = [
        "Glucose",
        "BloodPressure",
        "SkinThickness",
        "Insulin",
        "BMI"
    ]

    patient_data[columns_with_zero_as_missing] = (
        patient_data[
            columns_with_zero_as_missing
        ].replace(0, float("nan"))
    )


    # ========================================
    # PREPROCESS INPUT
    # ========================================

    patient_data_processed = preprocessor.transform(
        patient_data
    )


    # ========================================
    # PREDICTION
    # ========================================

    prediction = model.predict(
        patient_data_processed
    )[0]

    probability = model.predict_proba(
        patient_data_processed
    )[0][1]

    probability_percent = probability * 100


    # ========================================
    # RESULT
    # ========================================

    st.divider()

    st.subheader("📊 Prediction Result")


    if prediction == 1:

        st.error(
            "⚠️ Higher Diabetes Risk Detected"
        )

    else:

        st.success(
            "✅ Lower Diabetes Risk"
        )


    # Probability
    st.write(
        f"### Estimated Diabetes Probability: "
        f"{probability_percent:.2f}%"
    )

    st.progress(
        float(probability)
    )


    # Risk interpretation
    if probability_percent >= 70:

        st.warning(
            "High estimated risk. Please consult "
            "a qualified healthcare professional."
        )

    elif probability_percent >= 40:

        st.warning(
            "Moderate estimated risk. Consider "
            "discussing your results with a healthcare professional."
        )

    else:

        st.info(
            "Lower estimated risk based on the "
            "model prediction."
        )


    # ========================================
    # MODEL INFORMATION
    # ========================================

    st.divider()

    st.subheader("🤖 Model Information")

    col1, col2 = st.columns(2)

    with col1:
        st.metric(
            "Model",
            "XGBoost"
        )

    with col2:
        st.metric(
            "Test Accuracy",
            "75.97%"
        )


    # ========================================
    # DISCLAIMER
    # ========================================

    st.divider()

    st.warning(
        "⚠️ Educational Use Only: This application "
        "provides a machine-learning estimate and "
        "is not a medical diagnosis. Please consult "
        "a qualified healthcare professional for "
        "medical advice."
    )