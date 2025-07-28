import streamlit as st
import pandas as pd
import xgboost as xgb
import numpy as np
import pickle

# Load the trained model
model = pickle.load(open('xgb_attrition_model.pkl', 'rb'))

# Set Streamlit page config
st.set_page_config(page_title="Employee Attrition Predictor", layout="centered")

st.title("🧠 Employee Attrition Prediction")
st.markdown("This app predicts whether an employee is likely to leave the company based on their profile.")

# User input form
with st.form("prediction_form"):
    st.subheader("🔍 Enter Employee Details:")

    Age = st.slider("Age", 18, 60, 30)
    DistanceFromHome = st.slider("Distance From Home (in km)", 1, 30, 10)
    JobLevel = st.selectbox("Job Level", [1, 2, 3, 4, 5])
    MonthlyIncome = st.number_input("Monthly Income", min_value=1000, max_value=20000, value=5000, step=500)
    OverTime = st.selectbox("Does the employee work overtime?", ['Yes', 'No'])
    JobSatisfaction = st.selectbox("Job Satisfaction (1=Low, 4=High)", [1, 2, 3, 4])
    EnvironmentSatisfaction = st.selectbox("Environment Satisfaction (1=Low, 4=High)", [1, 2, 3, 4])
    WorkLifeBalance = st.selectbox("Work-Life Balance (1=Bad, 4=Excellent)", [1, 2, 3, 4])
    TotalWorkingYears = st.slider("Total Working Years", 0, 40, 10)
    YearsAtCompany = st.slider("Years at Company", 0, 30, 5)
    YearsInCurrentRole = st.slider("Years in Current Role", 0, 20, 3)
    YearsSinceLastPromotion = st.slider("Years Since Last Promotion", 0, 15, 2)

    submit = st.form_submit_button("🔮 Predict Attrition")

if submit:
    # Convert inputs into model-ready format
    input_data = pd.DataFrame({
        'Age': [Age],
        'DistanceFromHome': [DistanceFromHome],
        'JobLevel': [JobLevel],
        'MonthlyIncome': [MonthlyIncome],
        'OverTime': [1 if OverTime == 'Yes' else 0],
        'JobSatisfaction': [JobSatisfaction],
        'EnvironmentSatisfaction': [EnvironmentSatisfaction],
        'WorkLifeBalance': [WorkLifeBalance],
        'TotalWorkingYears': [TotalWorkingYears],
        'YearsAtCompany': [YearsAtCompany],
        'YearsInCurrentRole': [YearsInCurrentRole],
        'YearsSinceLastPromotion': [YearsSinceLastPromotion]
    })

    prediction = model.predict(input_data)
    prediction_proba = model.predict_proba(input_data)[0][1]

    if prediction[0] == 1:
        st.error(f"⚠️ High Risk of Attrition\n\nProbability: {prediction_proba:.2f}")
    else:
        st.success(f"✅ Low Risk of Attrition\n\nProbability: {prediction_proba:.2f}")
