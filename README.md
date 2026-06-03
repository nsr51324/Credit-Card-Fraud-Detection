# Credit Card Fraud Detection using Stacking Ensemble

## Overview
This project detects fraudulent credit card transactions using machine learning.
It handles highly imbalanced data using SMOTE and improves performance using a stacking ensemble model.

## Dataset
- Source: Credit Card Fraud Dataset
- Features: 30 numerical features + Time + Amount
- Target: Class (0 = Normal, 1 = Fraud)

## Workflow

### 1. Data Preprocessing
- Checked missing values
- Scaled `Time` and `Amount` using StandardScaler
- Removed NaN values

### 2. Handling Imbalanced Data
- Used SMOTE to balance classes

### 3. Model Architecture
Stacking Classifier:
- Base Models:
  - XGBoost Classifier
  - Random Forest Classifier
- Final Model:
  - Logistic Regression

## Evaluation
- Confusion Matrix
- Precision / Recall / F1-score
- Accuracy Score

## Results
- High recall on fraud detection class
- Improved performance using ensemble learning

## Model Saving
Saved using Joblib:
- fraud_model.pkl → trained stacking model
- scaler.pkl → StandardScaler

## Installation

```bash
pip install -r requirements.txt
