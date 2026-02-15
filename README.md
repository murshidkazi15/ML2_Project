# Student Dropout Prediction & Academic Risk Analytics

A Machine Learning project for predicting student academic dropout risk and identifying key behavioral and academic indicators using classification models.

---

## 📌 Project Overview

Student attrition is a critical challenge in higher education. Early identification of students facing academic or personal challenges enables institutions to intervene with targeted support. 

This project analyzes a dataset of **10,000 student records** containing academic performance metrics, demographic indicators, stress levels, and attendance rates to build predictive machine learning models for early dropout risk detection.

---

## 📊 Dataset Overview

- **Sample Size:** 10,000 instances
- **Target Variable:** `Dropout` (0 = Retained/Graduated, 1 = Dropped Out)
- **Key Features:**
  - **Academic:** `GPA`, `Semester_GPA`, `CGPA`, `Assignment_Delay_Days`, `Department`
  - **Behavioral & Personal:** `Attendance_Rate`, `Study_Hours_per_Day`, `Stress_Index`
  - **Socioeconomic:** `Family_Income`, `Scholarship`, `Part_Time_Job`, `Parental_Education`

---

## ⚙️ Methodology & Pipeline

1. **Exploratory Data Analysis (EDA):** Identified feature distributions, correlations, and missing values.
2. **Data Preprocessing:**
   - One-hot encoding for categorical variables.
   - Median imputation (`SimpleImputer`) for missing features.
   - Feature scaling (`StandardScaler`) for distance-sensitive models.
3. **Model Selection & Training:** Evaluated multiple supervised classification algorithms:
   - Logistic Regression
   - Random Forest Classifier
   - HistGradientBoosting Classifier
4. **Evaluation:** Stratified train-test split (80/20) evaluated using **Accuracy** and **ROC-AUC**.

---

## 🏆 Model Performance Results

| Model | Accuracy | ROC-AUC Score |
| :--- | :---: | :---: |
| **Logistic Regression** | **81.55%** | **0.8206** |
| **Random Forest Classifier** | 80.15% | 0.8053 |
| **HistGradientBoosting** | 79.65% | 0.7994 |

---

## 🔍 Key Risk Drivers

Feature importance analysis reveals the top factors contributing to student dropout risk:

1. **GPA & CGPA Performance:** Strongest negative correlation with dropout likelihood.
2. **Attendance Rate:** High absenteeism is a primary early warning signal.
3. **Stress Index:** Higher reported stress levels correlate directly with increased risk of attrition.
4. **Assignment Delay Days:** Consistent delays indicate declining engagement.

---

## 🚀 Getting Started

### Prerequisites
Ensure Python 3.8+ and required packages are installed:

```bash
pip install pandas numpy scikit-learn
```

### Running the Analysis
To train the models and output evaluation metrics, run:

```bash
python train_model.py
```

---

## 👤 Author

- **Murshid Kazi** — *Data Science & Information Management Student at NOVA IMS*
- GitHub: [@murshidkazi15](https://github.com/murshidkazi15)
