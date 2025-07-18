import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import pytz
import jdatetime
import time

# ============================
# ❤️ Heart Disease Risk Checker — Stylish & Improved UI
# ============================

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Heart Disease Risk Checker",
    page_icon="❤️",
    layout="centered"
)

# --- Load Models and Preprocessors ---
# Define the directory for saved models
MODEL_DIR = "saved_models"

# Dictionary to store machine learning models
MODELS = {
    "Support Vector Machine (SVM)": joblib.load(os.path.join(MODEL_DIR, "svm.pkl")),
    "Naive Bayes": joblib.load(os.path.join(MODEL_DIR, "naive_bayes.pkl")),
    "Logistic Regression": joblib.load(os.path.join(MODEL_DIR, "logistic_regression.pkl")),
    "Multi-Layer Perceptron (MLP)": joblib.load(os.path.join(MODEL_DIR, "mlp.pkl")),
    "Decision Tree": joblib.load(os.path.join(MODEL_DIR, "decision_tree.pkl")),
    "K-Nearest Neighbors (KNN)": joblib.load(os.path.join(MODEL_DIR, "knn.pkl")),
    "Random Forest": joblib.load(os.path.join(MODEL_DIR, "random_forest.pkl")),
    "XGBoost": joblib.load(os.path.join(MODEL_DIR, "xgboost.pkl")),
    "AdaBoost": joblib.load(os.path.join(MODEL_DIR, "adaboost.pkl")),
    "LightGBM": joblib.load(os.path.join(MODEL_DIR, "lightgbm.pkl")),
    "Voting Ensemble": joblib.load(os.path.join(MODEL_DIR, "voting.pkl")),
    "Stacking Ensemble": joblib.load(os.path.join(MODEL_DIR, "stacking.pkl"))
}

# Load scaler and encoder columns
SCALER = joblib.load(os.path.join(MODEL_DIR, "scaler.pkl"))
ENCODER_COLUMNS = joblib.load(os.path.join(MODEL_DIR, "encoder_columns.pkl"))

# --- CSS Styling ---
# Custom CSS for enhanced UI styling
st.markdown("""
<style>
    /* Date-Time Bar */
    .date-time-bar {
        background: #222;
        padding: 5px 10px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.2);
        color: #bbb;
        font-size: 0.85rem;
    }

    /* Table Styling */
    .styled-table {
        width: 100%;
        border-collapse: collapse;
        font-family: 'Roboto', sans-serif;
        font-size: 16px;
        margin-top: 1rem;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.3);
    }
    .styled-table th {
        background-color: #1A202C;
        color: #F26A6A;
        font-weight: 700;
        padding: 14px 16px;
        text-align: center;
        vertical-align: middle;
        letter-spacing: 0.4px;
    }
    .styled-table td {
        background-color: #1F2937;
        color: #E5E7EB;
        padding: 12px 16px;
        text-align: center;
        vertical-align: middle;
    }
    .styled-table tr:nth-child(even) {
        background-color: #2A2D3A;
    }
    .styled-table tr:nth-child(odd) {
        background-color: #1F2937;
    }
    .styled-table td[style*="color:#2ecc71"] {
        color: #27AE60;
    }
    .styled-table td[style*="color:#e74c3c"] {
        color: #E74C3C;
    }

    /* Unified Tooltip */
    .tooltip {
        display: inline-block;
        background-color: #A0AEC0;
        border-radius: 50%;
        width: 16px;
        height: 16px;
        color: #1A202C;
        font-size: 12px;
        line-height: 16px;
        text-align: center;
        margin-left: 6px;
        cursor: help;
        font-weight: bold;
    }

    /* Footer Styling */
    .footer {
        margin-top: 2rem;
        padding-top: 1rem;
        text-align: center;
        font-size: 0.85rem;
        color: #999;
        border-top: 1px solid #555;
        animation: pulse 2s infinite;
    }
    @keyframes pulse {
        0% { color: #999; }
        50% { color: #bbb; }
        100% { color: #999; }
    }

    /* Form Label Styling */
    div[data-baseweb="form-control"] > label {
        font-size: 15px !important;
        font-weight: 700 !important;
        color: #eeeeee !important;
        margin-bottom: 6px !important;
        letter-spacing: 0.3px;
    }

    /* Input Value Styling */
    .stTextInput input,
    .stNumberInput input,
    .stSelectbox div[data-baseweb="select"] > div,
    .stSlider {
        font-size: 13px !important;
        font-weight: 500 !important;
        color: #cccccc !important;
    }

    /* Dropdown Menu Option Styling */
    div[data-baseweb="menu"] * {
        font-size: 10px !important;
        font-weight: 400 !important;
        color: #ccc !important;
        padding: 6px 10px !important;
    }
</style>
""", unsafe_allow_html=True)

# --- Date-Time Display (Shamsi + Tehran Time) ---
# Get current time in Tehran timezone
now_tehran = datetime.now(pytz.timezone("Asia/Tehran"))
# Convert Gregorian date to Shamsi (Persian) date
shamsi_date = jdatetime.date.fromgregorian(date=now_tehran.date()).strftime('%Y/%m/%d')
time_str = now_tehran.strftime('%H:%M:%S')
st.markdown(
    f"""
    <div class='date-time-bar' style='display:flex; justify-content:space-between; align-items:center; margin-top:-1rem; margin-bottom:1rem;'>
        <div style='font-family:monospace;'>🕒 {shamsi_date} — {time_str}</div>
        <div style='margin-left:8px;'><img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/Flag_of_Iran.svg" width="32" style="border-radius:4px;" /></div>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Title with Animated Heart ---
st.markdown(
    """
    <h1 style='text-align:center;'>
        <span class='heartbeat' style='font-size: 1.8em;'>❤️</span> Heart Disease Risk Checker
    </h1>
    <style>
    .heartbeat {
        display: inline-block;
        animation: beat 1.2s infinite;
    }
    @keyframes beat {
        0%, 100% { transform: scale(1); }
        50% { transform: scale(1.3); }
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.markdown("Use this tool to estimate your **risk of heart disease** based on medical features.")

# --- Custom Selectbox Function ---
def custom_selectbox(label, options, key=None, help_text=""):
    """
    Custom selectbox with placeholder and help tooltip.
    
    Args:
        label (str): Label for the selectbox
        options (list): List of options for the selectbox
        key (str): Unique key for the widget
        help_text (str): Tooltip text for the selectbox
    
    Returns:
        Selected option or None
    """
    return st.selectbox(
        label=label,
        options=options,
        index=None,
        placeholder="Select...",
        key=key,
        help=help_text
    )

# --- Input Form ---
with st.form("user_input"):
    st.subheader("🩺 Enter Patient Clinical Data")
    col1, col2 = st.columns(2)

    with col1:
        # Slider for age input
        age = st.slider("Age", 20, 90, help="Patient's age in years")
        # Dropdown for biological sex
        sex = custom_selectbox("Sex", ["Male", "Female"], key="sex", help_text="Biological sex of the patient")
        # Dropdown for chest pain type
        cp = custom_selectbox(
            "Chest Pain Type",
            ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"],
            key="cp",
            help_text="Type of chest pain experienced"
        )
        # Input for resting blood pressure
        trestbps = st.number_input(
            "Resting Blood Pressure (mm Hg)",
            min_value=80,
            max_value=200,
            help="Resting blood pressure in mm Hg"
        )
        # Input for cholesterol level
        chol = st.number_input(
            "Serum Cholesterol (mg/dl)",
            min_value=100,
            max_value=600,
            help="Serum cholesterol level in mg/dl"
        )
        # Dropdown for fasting blood sugar
        fbs = custom_selectbox(
            "Fasting Blood Sugar > 120 mg/dl?",
            ["True", "False"],
            key="fbs",
            help_text="Fasting blood sugar > 120 mg/dl"
        )
        # Dropdown for ECG results
        restecg = custom_selectbox(
            "Resting Electrocardiographic Results",
            ["Normal", "ST-T abnormality", "Left Ventricular Hypertrophy"],
            key="restecg",
            help_text="Resting ECG results"
        )

    with col2:
        # Input for maximum heart rate
        thalach = st.number_input(
            "Maximum Heart Rate Achieved (bpm)",
            min_value=70,
            max_value=210,
            help="Maximum heart rate achieved during exercise"
        )
        # Dropdown for exercise-induced angina
        exang = custom_selectbox(
            "Exercise Induced Angina",
            ["Yes", "No"],
            key="exang",
            help_text="Presence of exercise-induced angina"
        )
        # Slider for ST depression
        oldpeak = st.slider(
            "ST Depression Induced by Exercise",
            0.0,
            6.0,
            step=0.1,
            help="ST depression induced by exercise relative to rest"
        )
        # Dropdown for ST segment slope
        slope = custom_selectbox(
            "Slope of Peak Exercise ST Segment",
            ["Upsloping", "Flat", "Downsloping"],
            key="slope",
            help_text="Slope of the peak exercise ST segment"
        )
        # Dropdown for number of major vessels
        ca = custom_selectbox(
            "Number of Major Vessels Colored by Fluoroscopy",
            ["0", "1", "2", "3"],
            key="ca",
            help_text="Number of major vessels (0-3) colored by fluoroscopy"
        )
        # Dropdown for thalassemia
        thal = custom_selectbox(
            "Thalassemia",
            ["Normal", "Fixed Defect", "Reversible Defect"],
            key="thal",
            help_text="Thalassemia test result"
        )

    # Submit button for the form
    submitted = st.form_submit_button("🔧 Predict")

# --- Prediction Logic ---
if submitted:
    # Validate all inputs are provided
    required_fields = [sex, cp, fbs, restecg, exang, slope, thal, ca]
    if None in required_fields:
        st.error("❌ Please fill in all fields before prediction.")
    else:
        with st.spinner("🔍 Analyzing your data..."):
            # Simulate processing delay for better UX
            time.sleep(0.5)

            # Prepare input dictionary for model
            input_dict = {
                'age': age,
                'sex': 1 if sex == "Male" else 0,
                'cp': ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"].index(cp),
                'trestbps': trestbps,
                'chol': chol,
                'fbs': 1 if fbs == "True" else 0,
                'restecg': ["Normal", "ST-T abnormality", "Left Ventricular Hypertrophy"].index(restecg),
                'thalach': thalach,
                'exang': 1 if exang == "Yes" else 0,
                'oldpeak': oldpeak,
                'slope': ["Upsloping", "Flat", "Downsloping"].index(slope),
                'ca': int(ca),
                'thal': ["Normal", "Fixed Defect", "Reversible Defect"].index(thal)
            }
            
            # Convert input to DataFrame and encode
            input_df = pd.DataFrame([input_dict])
            input_encoded = pd.get_dummies(input_df)
            input_encoded = input_encoded.reindex(columns=ENCODER_COLUMNS, fill_value=0)
            X_scaled = SCALER.transform(input_encoded)

            # Model recall dictionary (hardcoded for simplicity)
            MODEL_RECALLS = {
                "Support Vector Machine (SVM)": 89.29,
                "Naive Bayes": 78.57,
                "Logistic Regression": 82.14,
                "Multi-Layer Perceptron (MLP)": 85.71,
                "Decision Tree": 78.57,
                "K-Nearest Neighbors (KNN)": 78.57,
                "Random Forest": 89.29,
                "XGBoost": 85.71,
                "AdaBoost": 89.29,
                "LightGBM": 89.29,
                "Voting Ensemble": 92.86,
                "Stacking Ensemble": 85.71
            }

            # Sort models by recall score
            sorted_models = sorted(MODELS.items(), key=lambda x: MODEL_RECALLS.get(x[0], 0), reverse=True)

            # Create display names with tooltips
            header_with_tooltip = (
                f"<span>Predictor <span class='tooltip' title='Sorted from best to worst model based on Recall score'>?</span></span>"
            )
            display_names = [
                f"<span>Predictor {chr(65+i)} <span class='tooltip' title='{model_name} (Recall: {MODEL_RECALLS[model_name]:.2f}%)'>?</span></span>"
                for i, (model_name, _) in enumerate(sorted_models)
            ]

            # Perform predictions
            results = []
            for i, (model_name, model) in enumerate(sorted_models):
                pred = model.predict(X_scaled)[0]
                result_label = (
                    f"<span style='color:#e74c3c; font-weight:bold;'>❗️ High Risk</span>"
                    if pred == 1 else
                    f"<span style='color:#2ecc71; font-weight:bold;'>✅ Low Risk</span>"
                )
                results.append({
                    header_with_tooltip: display_names[i],
                    "Risk Prediction": result_label
                })

            # Convert results to DataFrame
            df_predictions = pd.DataFrame(results)

        # Display prediction results
        st.subheader("📋 Prediction Results")
        st.markdown(
            df_predictions.to_html(escape=False, index=False, classes="styled-table"),
            unsafe_allow_html=True
        )

# --- Footer ---
st.markdown(
    """
    <div class="footer">
        ❤️ Heart Disease Risk Checker — Version 1.0.0<br>
        Developed by <a href="mailto:farzadmohseni@aut.ac.ir" target="_blank" style="color:#f26a6a; text-decoration:none;"><strong>Farzad Mohseni</strong></a> | © 2025
    </div>
    """,
    unsafe_allow_html=True
)
