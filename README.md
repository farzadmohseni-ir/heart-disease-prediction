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

## ⚙️ ML Algorithms Implemented

✅ This project compares and evaluates the following classification models:

- Support Vector Machine (SVM)
- Naïve Bayes
- Logistic Regression
- Decision Tree
- K-Nearest Neighbors (KNN)
- Neural Network (MLPClassifier)
- Random Forest
- XGBoost
- AdaBoost
- LightGBM
- Voting Classifier
- Stacking Classifier

---

## 📊 Results (Test Set)

| Model               | Accuracy | Precision | Recall | F1 Score |
|---------------------|----------|-----------|--------|----------|
| **SVM**             | 93.33%   | 96.15%    | 89.29% | 92.59%   |
| **Logistic Reg.**   | 91.67%   | 100.00%   | 82.14% | 90.20%   |
| **Stacking Class.** | 93.33%   | 100.00%   | 85.71% | 92.31%   |
| **Voting Classifier** | 90.00% | 86.67%    | 92.86% | 89.66%   |
| *...others included in full report* |

📌 **Conclusion:**  
The **Stacking Classifier** showed the best overall performance, combining high precision, recall, and F1 score.

