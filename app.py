
import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(
    page_title="Diabetes AI Dashboard",
    page_icon="🩺",
    layout="wide"
)

# ---------- CUSTOM THEME ----------
st.markdown("""
<style>
/* Main cream background */
.stApp {
    background: #FFF9EE;
    color: #263449;
}

.block-container {
    max-width: 1200px;
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* Make all normal text clearly visible */
.stApp p,
.stApp label,
.stApp .stMarkdown,
.stApp [data-testid="stCaptionContainer"] {
    color: #263449;
}

/* Main title banner */
.hero {
    background: linear-gradient(120deg, #1746A2, #633BB7, #087F8C);
    padding: 32px;
    border-radius: 22px;
    color: white !important;
    margin-bottom: 25px;
    box-shadow: 0 8px 24px rgba(35, 55, 100, 0.16);
}

.hero h1 {
    color: #FFFFFF !important;
    font-size: 35px;
    margin-bottom: 10px;
}

.hero p {
    color: #F5F7FF !important;
    font-size: 16px;
    margin-bottom: 5px;
}

/* Section headings */
.stApp h2,
.stApp h3,
.stApp h4 {
    color: #203B69 !important;
    font-weight: 700;
}

/* Information cards */
.info-card {
    background: #FFFFFF;
    border: 1px solid #E7DCC9;
    border-radius: 16px;
    padding: 20px;
    text-align: center;
    min-height: 105px;
    box-shadow: 0 4px 12px rgba(80, 65, 40, 0.06);
}

.info-card h3 {
    color: #254B91 !important;
    margin: 0;
    font-size: 23px;
}

.info-card p {
    color: #4B5568 !important;
    margin-top: 8px;
    font-size: 14px;
}

/* Input area */
div[data-testid="stSelectbox"],
div[data-testid="stNumberInput"] {
    background: #FFFDF9;
    border-radius: 10px;
}

/* Input labels */
.stSelectbox label,
.stNumberInput label {
    color: #203B69 !important;
    font-weight: 600 !important;
}

/* White input fields with dark text */
.stSelectbox div[data-baseweb="select"] > div {
    background: #FFFFFF;
    border-color: #C8D2E2;
    color: #243449;
}

.stNumberInput input {
    background: #FFFFFF !important;
    color: #243449 !important;
    border-color: #C8D2E2 !important;
}

/* Input selected values */
.stSelectbox [data-testid="stMarkdownContainer"] p {
    color: #243449 !important;
}

/* Primary buttons */
.stButton > button,
.stFormSubmitButton > button {
    width: 100%;
    min-height: 48px;
    border-radius: 12px;
    border: none;
    background: linear-gradient(90deg, #2459D3, #713BC1);
    color: #FFFFFF !important;
    font-weight: 700;
    font-size: 16px;
}

.stButton > button:hover,
.stFormSubmitButton > button:hover {
    color: #FFFFFF !important;
    border: 1px solid #2459D3;
    box-shadow: 0 5px 15px rgba(80, 65, 190, 0.2);
}

/* Form card */
div[data-testid="stForm"] {
    background: #FFFCF6;
    border: 1px solid #E7DCC9;
    border-radius: 18px;
    padding: 22px;
}

/* Prediction metrics */
div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #E7DCC9;
    padding: 18px;
    border-radius: 14px;
}

div[data-testid="stMetricLabel"] {
    color: #354765 !important;
}

div[data-testid="stMetricValue"] {
    color: #1746A2 !important;
}

/* Expandable section */
div[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #E7DCC9;
    border-radius: 12px;
}

/* Footer */
.footer {
    text-align: center;
    color: #4B5568 !important;
    font-size: 13px;
    padding: 22px 0 5px 0;
}

/* Horizontal divider */
hr {
    border-color: #E1D5C2;
}
</style>
""", unsafe_allow_html=True)

# ---------- LOAD MODEL ----------
MODEL_PATH = "models/diabetes_model_compressed.joblib"
PREPROCESSOR_PATH = "models/diabetes_preprocessor.joblib"

if not os.path.exists(MODEL_PATH) or not os.path.exists(PREPROCESSOR_PATH):
    st.error("Model files not found. Please run: python train.py")
    st.stop()

try:
    model = joblib.load(MODEL_PATH)
    preprocessor = joblib.load(PREPROCESSOR_PATH)
except Exception as e:
    st.error(f"Could not load model files: {e}")
    st.stop()

# ---------- HEADER ----------
st.markdown("""
<div class="hero">
    <h1>🩺 Diabetes Prediction Assistant</h1>
    <p>AI-powered diabetes risk estimation dashboard</p>
    <p>Enter your health information below to get a machine-learning estimate.</p>
</div>
""", unsafe_allow_html=True)

# ---------- INFORMATION CARDS ----------
c1, c2, c3 = st.columns(3)

with c1:
    st.markdown("""
    <div class="info-card">
        <h3>🤖 AI Powered</h3>
        <p>Machine-learning model</p>
    </div>
    """, unsafe_allow_html=True)

with c2:
    st.markdown("""
    <div class="info-card">
        <h3>👥 All Genders</h3>
        <p>Female, Male and Other</p>
    </div>
    """, unsafe_allow_html=True)

with c3:
    st.markdown("""
    <div class="info-card">
        <h3>📊 Risk Estimate</h3>
        <p>Based on entered details</p>
    </div>
    """, unsafe_allow_html=True)

st.write("")
st.subheader("📝 Enter Health Details")
st.caption("Complete all fields below to generate a prediction.")

# ---------- HEALTH FORM ----------
with st.form("diabetes_form"):

    st.markdown("### 👤 Personal Information")

    col1, col2 = st.columns(2)

    with col1:
        gender = st.selectbox(
            "Gender",
            ["Female", "Male", "Other"]
        )

    with col2:
        age = st.number_input(
            "Age (years)",
            min_value=0.0,
            max_value=120.0,
            value=30.0,
            step=1.0
        )

    st.markdown("---")
    st.markdown("### ❤️ Health Information")

    col1, col2 = st.columns(2)

    with col1:
        hypertension = st.selectbox(
            "Hypertension (high blood pressure)",
            ["No", "Yes"]
        )

        bmi = st.number_input(
            "BMI",
            min_value=10.0,
            max_value=100.0,
            value=25.0,
            step=0.1
        )

        hba1c_level = st.number_input(
            "HbA1c Level (%)",
            min_value=3.0,
            max_value=20.0,
            value=5.5,
            step=0.1
        )

    with col2:
        heart_disease = st.selectbox(
            "Heart Disease",
            ["No", "Yes"]
        )

        smoking_history = st.selectbox(
            "Smoking History",
            [
                "never",
                "No Info",
                "current",
                "former",
                "ever",
                "not current"
            ]
        )

        blood_glucose_level = st.number_input(
            "Blood Glucose Level (mg/dL)",
            min_value=50,
            max_value=500,
            value=100,
            step=1
        )

    submitted = st.form_submit_button(
        "🔍 Predict Diabetes Risk"
    )

# ---------- PREDICTION RESULT ----------
if submitted:
    input_data = pd.DataFrame([{
        "gender": gender,
        "age": age,
        "hypertension": 1 if hypertension == "Yes" else 0,
        "heart_disease": 1 if heart_disease == "Yes" else 0,
        "smoking_history": smoking_history,
        "bmi": bmi,
        "HbA1c_level": hba1c_level,
        "blood_glucose_level": blood_glucose_level
    }])

    try:
        processed_data = preprocessor.transform(input_data)
        prediction = model.predict(processed_data)[0]
        probability = model.predict_proba(processed_data)[0][1]

        st.markdown("---")
        st.subheader("📊 Prediction Results")

        result_col1, result_col2 = st.columns(2)

        with result_col1:
            st.metric(
                "Estimated Model Risk",
                f"{probability * 100:.1f}%"
            )

        with result_col2:
            st.metric(
                "Selected Gender",
                gender
            )

        st.progress(float(max(0, min(1, probability))))

        if prediction == 1:
            st.warning(
                "⚠️ The model flagged a higher likelihood of diabetes. "
                "Please consult a qualified healthcare professional."
            )
        else:
            st.success(
                "✅ The model did not flag a higher likelihood in this result. "
                "This does not rule out diabetes."
            )

        st.caption(
            "This percentage is a model estimate, not a confirmed diagnosis."
        )

        with st.expander("🔎 View entered health information"):
            st.dataframe(input_data, use_container_width=True)

    except Exception as e:
        st.error(f"Prediction error: {e}")

# ---------- FOOTER ----------
st.markdown("""
<div class="footer">
    🩺 Diabetes Prediction Assistant | Academic Machine Learning Project
    <br>
    This tool is for educational purposes only and is not a medical diagnosis.
</div>
""", unsafe_allow_html=True)