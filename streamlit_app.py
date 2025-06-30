import streamlit as st
import pandas as pd
import joblib

# Load model
model = joblib.load("best_final_model.pkl")

# Set page title
st.set_page_config(page_title="Loan Default Predictor", layout="centered")
st.title("🏦 Loan Default Risk Classifier")

# Form for input
with st.form("loan_form"):
    age = st.number_input("Age", value=30)
    experience = st.number_input("Experience", value=5)
    income = st.number_input("Income (in thousands)", value=50)
    zip_code = st.number_input("ZIP Code", value=12345)
    family = st.selectbox("Family Size", [1, 2, 3, 4])
    ccavg = st.number_input("Average Credit Card Spend (CCAvg)", value=1.5)
    education = st.selectbox("Education Level", [1, 2, 3], format_func=lambda x: ["Undergrad", "Graduate", "Advanced/Professional"][x-1])
    mortgage = st.number_input("Mortgage (in thousands)", value=0)
    securities = st.selectbox("Securities Account", [0, 1], format_func=lambda x: "Yes" if x else "No")
    cd = st.selectbox("CD Account", [0, 1], format_func=lambda x: "Yes" if x else "No")
    online = st.selectbox("Online Banking", [0, 1], format_func=lambda x: "Yes" if x else "No")
    creditcard = st.selectbox("Credit Card", [0, 1], format_func=lambda x: "Yes" if x else "No")

    submit = st.form_submit_button("🔍 Predict")

# If the form is submitted
if submit:
    try:
        # Create input DataFrame
        input_data = pd.DataFrame([{
            'Age': age,
            'Experience': experience,
            'Income': income,
            'ZIP Code': zip_code,
            'Family': family,
            'CCAvg': ccavg,
            'Education': education,
            'Mortgage': mortgage,
            'Securities Account': securities,
            'CD Account': cd,
            'Online': online,
            'CreditCard': creditcard
        }])

        # Reorder columns to match model
        expected_columns = model.feature_names_in_
        input_data = input_data[expected_columns]

        # Make prediction
        pred = model.predict(input_data)[0]
        prob = model.predict_proba(input_data)[0][1]

        # Show result
        result = "🟥 Defaulter" if pred == 1 else "🟩 Non-Defaulter"
        st.success(f"**Prediction:** {result}")
        st.info(f"**Probability of default:** {prob*100:.2f}%")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
