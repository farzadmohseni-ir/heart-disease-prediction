import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from datetime import datetime
import pytz
import jdatetime
import time
import streamlit.components.v1 as components

# تنظیم پیکربندی صفحه
st.set_page_config(page_title="Heart Disease Risk Checker", page_icon="❤️", layout="centered")

# تنظیم Viewport و مقیاس پویا
components.html("""
    <script>
        const meta = document.createElement('meta');
        meta.name = "viewport";
        const scale = Math.min(window.innerWidth / 1200, 1.0);
        meta.content = `width=1200, initial-scale=${scale}, maximum-scale=${scale}, user-scalable=no`;
        document.getElementsByTagName('head')[0].appendChild(meta);

        function adjustScale() {
            const scale = Math.min(window.innerWidth / 1200, 1.0);
            document.documentElement.style.setProperty('--scale', scale);
            document.querySelector('.stApp').style.width = '1200px';
        }
        adjustScale();
        window.addEventListener('resize', adjustScale);

        document.addEventListener('DOMContentLoaded', () => {
            const columns = document.querySelectorAll('.stColumns > div');
            columns.forEach(col => {
                col.style.display = 'flex';
                col.style.flexWrap = 'nowrap';
                col.querySelectorAll('div').forEach(child => {
                    child.style.width = '50%';
                    child.style.padding = '0 10px';
                });
            });
        });
    </script>
""", height=0)

# CSS برای حفظ ظاهر دسکتاپ
st.markdown("""
<style>
    .stApp {
        width: 1200px;
        margin: 0 auto;
        overflow-x: hidden;
    }
    .stColumns > div {
        display: flex !important;
        flex-wrap: nowrap !important;
    }
    .stColumns > div > div {
        width: 50% !important;
        padding: 0 10px;
    }
    .styled-table {
        width: 100%;
        max-width: 1200px;
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
    }
    .styled-table td {
        background-color: #1F2937;
        color: #E5E7EB;
        padding: 12px 16px;
        text-align: center;
    }
    .styled-table tr:nth-child(even) {
        background-color: #2A2D3A;
    }
    .styled-table td[style*="color:#2ecc71"] {
        color: #27AE60;
    }
    .styled-table td[style*="color:#e74c3c"] {
        color: #E74C3C;
    }
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
    .footer {
        margin-top: 2rem;
        padding-top: 1rem;
        text-align: center;
        font-size: 0.85rem;
        color: #999;
        border-top: 1px solid #555;
    }
    div[data-baseweb="form-control"] > label {
        font-size: 15px !important;
        font-weight: 700 !important;
        color: #eeeeee !important;
    }
    .stTextInput input, .stNumberInput input, .stSelectbox div[data-baseweb="select"] > div {
        font-size: 13px !important;
    }
    @media (max-width: 1200px) {
        .stApp {
            transform: scale(var(--scale, 1));
            transform-origin: top left;
            width: 1200px !important;
        }
        body {
            overflow-x: hidden;
        }
    }
</style>
""", unsafe_allow_html=True)

# بقیه کد شما (بدون تغییر)
# --- Load Models and Preprocessors ---
model_dir = "saved_models"
models = {
    "Support Vector Machine (SVM)": joblib.load(os.path.join(model_dir, "svm.pkl")),
    "Naive Bayes": joblib.load(os.path.join(model_dir, "naive_bayes.pkl")),
    "Logistic Regression": joblib.load(os.path.join(model_dir, "logistic_regression.pkl")),
    "Multi-Layer Perceptron (MLP)": joblib.load(os.path.join(model_dir, "mlp.pkl")),
    "Decision Tree": joblib.load(os.path.join(model_dir, "decision_tree.pkl")),
    "K-Nearest Neighbors (KNN)": joblib.load(os.path.join(model_dir, "knn.pkl")),
    "Random Forest": joblib.load(os.path.join(model_dir, "random_forest.pkl")),
    "XGBoost": joblib.load(os.path.join(model_dir, "xgboost.pkl")),
    "AdaBoost": joblib.load(os.path.join(model_dir, "adaboost.pkl")),
    "LightGBM": joblib.load(os.path.join(model_dir, "lightgbm.pkl")),
    "Voting Ensemble": joblib.load(os.path.join(model_dir, "voting.pkl")),
    "Stacking Ensemble": joblib.load(os.path.join(model_dir, "stacking.pkl"))
}
scaler = joblib.load(os.path.join(model_dir, "scaler.pkl"))
encoder_columns = joblib.load(os.path.join(model_dir, "encoder_columns.pkl"))

# --- Date-Time Display (Shamsi + Tehran Time) ---
now_tehran = datetime.now(pytz.timezone("Asia/Tehran"))
shamsi_date = jdatetime.date.fromgregorian(date=now_tehran.date()).strftime('%Y/%m/%d')
time_str = now_tehran.strftime('%H:%M:%S')
st.markdown(f"""
<div class='date-time-bar' style='display:flex; justify-content:space-between; align-items:center; margin-top:-1rem; margin-bottom:1rem;'>
    <div style='font-family:monospace;'>🕒 {shamsi_date} — {time_str}</div>
    <div style='margin-left:8px;'><img src="https://upload.wikimedia.org/wikipedia/commons/c/ca/Flag_of_Iran.svg" width="32" style="border-radius:4px;" /></div>
</div>
""", unsafe_allow_html=True)

# --- Title with Animated Heart ---
st.markdown("""
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
""", unsafe_allow_html=True)
st.markdown("Use this tool to estimate your **risk of heart disease** based on medical features.")

# --- Custom Selectbox Function ---
def custom_selectbox(label, options, key=None, help_text=""):
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
        age = st.slider("Age", 20, 90, help="Patient's age")
        sex = custom_selectbox("Sex", ["Male", "Female"], key="sex", help_text="sex")
        cp = custom_selectbox("Chest Pain Type", 
                              ["Typical Angina", "Atypical Angina", "Non-Anginal Pain", "Asymptomatic"], 
                              key="cp", help_text="cp")
        trestbps = st.number_input("Resting Blood Pressure (mm Hg)", min_value=80, max_value=200, help="trestbps")
        chol = st.number_input("Serum Cholesterol (mg/dl)", min_value=100, max_value=600, help="chol")
        fbs = custom_selectbox("Fasting Blood Sugar > 120 mg/dl?", ["True", "False"], key="fbs", help_text="fbs")
        restecg = custom_selectbox("Resting Electrocardiographic Results", 
                                   ["Normal", "ST-T abnormality", "Left Ventricular Hypertrophy"], 
                                   key="restecg", help_text="restecg")

    with col2:
        thalach = st.number_input("Maximum Heart Rate Achieved (bpm)", min_value=70, max_value=210, help="thalach")
        exang = custom_selectbox("Exercise Induced Angina", ["Yes", "No"], key="exang", help_text="exang")
        oldpeak = st.slider("ST Depression Induced by Exercise", 0.0, 6.0, step=0.1, help="oldpeak")
        slope = custom_selectbox("Slope of Peak Exercise ST Segment", 
                                 ["Upsloping", "Flat", "Downsloping"], 
                                 key="slope", help_text="slope")
        ca = custom_selectbox("Number of Major Vessels Colored by Fluoroscopy", ["0", "1", "2", "3"], key="ca", help_text="ca")
        thal = custom_selectbox("Thalassemia", 
                                ["Normal", "Fixed Defect", "Reversible Defect"], 
                                key="thal", help_text="thal")

    submitted = st.form_submit_button("🔧 Predict")

# --- Prediction Logic ---
if submitted:
    # Validation
    if None in [sex, cp, fbs, restecg, exang, slope, thal, ca]:
        st.error("❌ Please fill in all fields before prediction.")
    else:
        with st.spinner("🔍 Analyzing your data..."):
            time.sleep(0.5)

            # Prepare Input
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
            input_df = pd.DataFrame([input_dict])
            input_encoded = pd.get_dummies(input_df)
            input_encoded = input_encoded.reindex(columns=encoder_columns, fill_value=0)
            X_scaled = scaler.transform(input_encoded)

            # Model Recall Dictionary
            model_recalls = {
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

            # Sort Models by Recall
            sorted_models = sorted(models.items(), key=lambda x: model_recalls.get(x[0], 0), reverse=True)

            # Table Header with Tooltip
            header_with_tooltip = (
                f"<span>Predictor <span class='tooltip' title='Sorted from best to worst model based on Recall score'>?</span></span>"
            )
            display_names = [
                f"<span>Predictor {chr(65+i)} <span class='tooltip' title='{model_name} (Recall: {model_recalls[model_name]:.2f})'>?</span></span>"
                for i, (model_name, _) in enumerate(sorted_models)
            ]

            # Prediction Results
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

            df_predictions = pd.DataFrame(results)

        # Show Results Table after prediction
        st.subheader("📋 Prediction Results")
        st.markdown(df_predictions.to_html(escape=False, index=False, classes="styled-table"), unsafe_allow_html=True)

# --- Footer ---
st.markdown("""
<div class="footer">
    ❤️ Heart Disease Risk Checker — Version 1.0.0<br>
    Developed by <a href="mailto:farzadmohseni@aut.ac.ir" target="_blank" style="color:#f26a6a; text-decoration:none;"><strong>Farzad Mohseni</strong></a> | © 2025
</div>
""", unsafe_allow_html=True)
