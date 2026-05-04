import streamlit as st
import joblib
import pandas as pd
import numpy as np

# 1. Load Assets
@st.cache_resource
def load_assets():
    model = joblib.load('credit_risk_model.pkl')
    cols = joblib.load('model_columns.pkl')
    return model, cols

model, model_columns = load_assets()

# 2. UI Header
st.set_page_config(page_title="FinGuard AI", layout="centered")
st.title("💳 FinGuard AI - Credit Risk Predictor")
st.markdown("Predict loan default probability using Machine Learning")
st.divider()

# 3. User Inputs
col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", 18, 90, 25)
    income = st.number_input("Annual Income ($)", 0, 1000000, 50000)
    home_ownership = st.selectbox("Home Ownership", ["RENT", "OWN", "MORTGAGE", "OTHER"])
    intent = st.selectbox("Loan Intent", ["PERSONAL", "EDUCATION", "MEDICAL", "VENTURE", "HOMEIMPROVEMENT", "DEBTCONSOLIDATION"])

with col2:
    emp_length = st.number_input("Years of Employment", 0, 50, 2)
    loan_amount = st.number_input("Loan Amount ($)", 0, 500000, 10000)
    grade = st.selectbox("Loan Grade", ["A", "B", "C", "D", "E", "F", "G"])
    int_rate = st.slider("Interest Rate (%)", 5.0, 25.0, 11.0)

# 4. The Prediction "Bridge"
if st.button("Analyze Risk"):
    # Create a template DataFrame with all 23 columns set to 0
    input_df = pd.DataFrame(np.zeros((1, len(model_columns))), columns=model_columns)
    
    # Fill numeric values
    input_df['person_age'] = age
    input_df['person_income'] = income
    input_df['person_emp_length'] = emp_length
    input_df['loan_amnt'] = loan_amount
    input_df['loan_int_rate'] = int_rate
    
    # Handle Categorical logic (One-Hot Encoding manually)
    # For example, if user picks RENT, we set 'person_home_ownership_RENT' to 1
    if f"person_home_ownership_{home_ownership}" in model_columns:
        input_df[f"person_home_ownership_{home_ownership}"] = 1
        
    if f"loan_intent_{intent}" in model_columns:
        input_df[f"loan_intent_{intent}"] = 1
        
    if f"loan_grade_{grade}" in model_columns:
        input_df[f"loan_grade_{grade}"] = 1

    # 5. Make Prediction
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[0][1] # Probability of default
    
    st.subheader("📋 Input Summary")
    st.write(input_df)


    # 6. Display Result
    st.divider()
    st.subheader("📊 Prediction Result")

    st.write(f"**Default Probability:** {probability:.2%}")

    if prediction[0] == 0:
        st.success("✅ Loan Approved (Low Risk)")
    else:
        st.error("❌ Loan Rejected (High Risk)")

    # 7. Risk Level Indicator
    if probability < 0.3:
        st.info("🟢 Low Risk")
    elif probability < 0.6:
        st.warning("🟡 Medium Risk")
    else:
        st.error("🔴 High Risk")