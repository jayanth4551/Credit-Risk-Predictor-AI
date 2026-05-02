# 🏦 FinGuard: AI-Powered Credit Risk Assessment

An end-to-end Machine Learning web application designed to predict the likelihood of loan default. This project demonstrates a full data science pipeline: from Exploratory Data Analysis (EDA) and handling imbalanced data to deploying a functional UI.

## 🚀 Live Demo

> [View Live App.. Coming Soon..](https://your-app-link.streamlit.app)

## 📊 Project Overview
Credit risk is a critical challenge for financial institutions. This project uses a **Random Forest Classifier** to assess whether a loan applicant is likely to default based on features like income, age, loan amount, and historical credit behavior.

### Key Highlights:
* **Accuracy:** 92%
* **F1-Score (Class 1):** 0.79 (Strong performance on minority default class)
* **Sampling:** Utilized **SMOTE** to address class imbalance.
* **Cleaning:** Handled outliers (e.g., impossible ages) and missing values via median imputation.

## 🛠️ Tech Stack
* **Language:** Python 3.13
* **Machine Learning:** Scikit-Learn, Imbalanced-Learn
* **Data Handling:** Pandas, NumPy
* **Visualization:** Matplotlib, Seaborn
* **Frontend:** Streamlit

## 📁 Repository Structure
```text
├── app.py                   # Streamlit web application
├── credit_risk_model.pkl    # Pre-trained Random Forest model
├── model_columns.pkl        # List of features for data alignment
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation