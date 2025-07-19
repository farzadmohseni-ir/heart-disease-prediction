# heart-disease-prediction
Predicting heart disease using various machine learning algorithms and the UCI Cleveland dataset.


# 💓 Heart Disease Prediction using Machine Learning

A machine learning project to predict heart disease based on clinical features using various classification algorithms. This implementation is inspired by the paper:  
📄 **“Prediction of Heart Disease UCI Dataset Using Machine Learning Algorithms”** (EMACS Journal, 2022)  

---

## 📚 Dataset

- **Name:** UCI Cleveland Heart Disease Dataset  
- **Source:** 
  - [UCI Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
  - [Kaggle Link](https://www.kaggle.com/datasets/cherngs/heart-disease-cleveland-uci)
- **Records:** 297 patients  
- **Features:** 13 clinical features + 1 target label

### 🔍 Features Used:

| Feature       | Description                                                                 |
|---------------|-----------------------------------------------------------------------------|
| `age`         | Age in years                                                                |
| `sex`         | Gender (1 = male; 0 = female)                                               |
| `cp`          | Chest pain type (0–3)                                                       |
| `trestbps`    | Resting blood pressure (mm Hg)                                              |
| `chol`        | Serum cholesterol (mg/dl)                                                   |
| `fbs`         | Fasting blood sugar > 120 mg/dl (1 = true; 0 = false)                       |
| `restecg`     | Resting ECG results (0–2)                                                   |
| `thalach`     | Maximum heart rate achieved                                                 |
| `exang`       | Exercise-induced angina (1 = yes; 0 = no)                                   |
| `oldpeak`     | ST depression induced by exercise                                           |
| `slope`       | Slope of the peak exercise ST segment (0–2)                                 |
| `ca`          | Number of major vessels colored by fluoroscopy (0–3)                        |
| `thal`        | Thalassemia (0 = normal; 1 = fixed defect; 2 = reversible defect)           |
| `condition`   | Target class (0 = no disease; 1 = heart disease)                            |

---




---

## 🧮 Scientific Scoring Formula for Medical Models

In clinical prediction tasks, minimizing false negatives is crucial. Therefore, we use a weighted formula to evaluate models more scientifically:

**Scientific Score** = 0.35 × Recall + 0.30 × F1 Score + 0.15 × Specificity + 0.10 × Precision + 0.10 × Accuracy


<div style="display: flex; justify-content: space-between; gap: 20px; flex-wrap: wrap;">

  <div style="flex: 1; min-width: 300px; text-align: center;">
    <img src="images/rank models.jpg" alt="Scientific Model Ranking" style="width: 100%; max-width: 500px; border-radius: 8px;">
  </div>
<\b>
  <div style="flex: 1; min-width: 300px; text-align: center;">
    <img src="images/test models.jpg" alt="Test Set Metrics Summary" style="width: 100%; max-width: 500px; border-radius: 8px;">
  </div>

</div>







---
## ⚙️ ML Algorithms Implemented & Results (Test Set)


| Model               | Accuracy | Precision | Recall | Specificity | F1 Score |
|---------------------|----------|-----------|--------|-------------|----------|
| **SVM**             | 93.33%   | 96.15%    | 89.29% | 96.88%      | 92.59%   |
| **Naive Bayes**     | 85.00%   | 88.00%    | 78.57% | 90.62%      | 83.02%   |
| **Logistic Regression** | 91.67% | 100.00%   | 82.14% | 100.00%     | 90.20%   |
| **Neural Network**  | 90.00%   | 92.31%    | 85.71% | 93.75%      | 88.89%   |
| **Decision Tree**   | 76.67%   | 73.33%    | 78.57% | 75.00%      | 75.86%   |
| **KNN**             | 86.67%   | 91.67%    | 78.57% | 93.75%      | 84.62%   |
| **Random Forest**   | 83.33%   | 78.12%    | 89.29% | 78.12%      | 83.33%   |
| **XGBoost**         | 83.33%   | 88.89%    | 85.71% | 90.62%      | 87.27%   |
| **AdaBoost**        | 86.67%   | 83.33%    | 89.29% | 84.38%      | 86.21%   |
| **LightGBM**        | 85.00%   | 80.65%    | 89.29% | 81.25%      | 84.75%   |
| **Voting Classifier** | 90.00% | 86.67%    | 92.86% | 87.50%      | 89.66%   |
| **Stacking Classifier** | 93.33% | 100.00%   | 85.71% | 100.00%     | 92.31%   |

---

## ▶️ How to Run the Project

You can run this heart disease prediction app in **two ways**:


### 🖥️ Option 1: Run Locally (Recommended for Developers)

> ✅ **Requirements**  
> - Python: `3.11.13`  
> - Dependencies: Listed in `requirements.txt`  
> - OS: Windows, macOS, or Linux

### 📌 Steps:

```bash
# 1. Clone the repository
git clone https://github.com/your-username/heart-disease-prediction.git
cd heart-disease-prediction

# 2. (Optional) Create and activate a virtual environment
python -m venv venv

# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate

# 3. Install required dependencies
pip install -r requirements.txt

# 4. Run the Streamlit app
streamlit run app.py


The app will launch at:  
👉 `http://localhost:8501`

```


### 🌐 Option 2: Run Online (Zero Setup)

You can try the app instantly via Streamlit Cloud:

🔗 **[Launch App Online](https://heart-disease-prediction-with-farzad-mohseni.streamlit.app/)**

> ⚠️ **Important Notes:**
> - Please ensure your **VPN is active** if access is restricted in your region.
> - Use **Desktop View** for proper layout and display.

---

## ⚠️ Limitations

This project, while demonstrating the potential of machine learning for heart disease prediction, has several limitations:

- 📊 **Small dataset**: The UCI Cleveland dataset includes only 297 records, which may lead to overfitting and limit the model's generalizability to diverse populations.
- 🌍 **Geographical bias**: The dataset is based on patients from a specific region, potentially reducing the model's accuracy for global or demographically different populations.
- 🩺 **Limited clinical features**: Only 13 features are used, whereas real-world diagnostics often include additional data like imaging or advanced biomarkers.
- 🧪 **Lack of external validation**: The model has not been tested on independent or real-time clinical data, limiting its real-world applicability.

📌 *This project is for educational and experimental purposes only and should not be used for medical decision-making.*

---

## 🤝 Contributions

We welcome contributions to make this project even better! 🎉 If you'd like to contribute, here are some ideas:

- Enhance the model or training pipeline (e.g., hyperparameter tuning, new algorithms)
- Add support for larger or more diverse datasets
- Improve the UI/UX of the Streamlit app
- Fix bugs or optimize performance
- Translate the app into other languages

📬 Have a new feature idea or want to collaborate on expanding the dataset? Feel free to reach out directly!
