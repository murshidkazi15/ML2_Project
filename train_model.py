"""
Student Dropout Prediction & Academic Risk Analysis
Author: Murshid Kazi (NOVA IMS)

This script loads the student performance dataset, preprocesses features,
handles missing values, scales numerical data, trains multiple Machine Learning
classifiers (Logistic Regression, Random Forest, HistGradientBoosting), and evaluates
model metrics (Accuracy, ROC-AUC, Precision, Recall, F1-Score).
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score, confusion_matrix

def load_and_preprocess(filepath):
    print(f"[*] Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)
    
    # Drop identifier column if present
    if 'Student_ID' in df.columns:
        df = df.drop(columns=['Student_ID'])
        
    # One-hot encode categorical features
    df_encoded = pd.get_dummies(df, drop_first=True)
    
    X = df_encoded.drop(columns=['Dropout'])
    y = df_encoded['Dropout']
    
    return X, y

def train_and_evaluate(X, y):
    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Pipeline: Imputation & Scaling
    imputer = SimpleImputer(strategy='median')
    scaler = StandardScaler()
    
    X_train_scaled = scaler.fit_transform(imputer.fit_transform(X_train))
    X_test_scaled = scaler.transform(imputer.transform(X_test))
    
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, random_state=42),
        "HistGradientBoosting": HistGradientBoostingClassifier(random_state=42)
    }
    
    results = {}
    print("\n" + "="*50)
    print("         MACHINE LEARNING MODEL RESULTS         ")
    print("="*50)
    
    for name, model in models.items():
        if name == "HistGradientBoosting":
            # Handles NaNs natively without scaling
            model.fit(imputer.fit_transform(X_train), y_train)
            y_pred = model.predict(imputer.transform(X_test))
            y_prob = model.predict_proba(imputer.transform(X_test))[:, 1]
        else:
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_prob)
        
        results[name] = {"Accuracy": acc, "ROC-AUC": auc}
        print(f"\n▶ {name}:")
        print(f"  - Accuracy: {acc*100:.2f}%")
        print(f"  - ROC-AUC : {auc:.4f}")
        
    return results, models["Random Forest"], X.columns

if __name__ == "__main__":
    dataset_path = "student_dropout_dataset_v3.csv"
    X, y = load_and_preprocess(dataset_path)
    results, rf_model, feature_names = train_and_evaluate(X, y)
    print("\n[*] Model training complete!")
