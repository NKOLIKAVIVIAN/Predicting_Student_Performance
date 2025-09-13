import streamlit as st
import pandas as pd
import joblib

# Load model and scaler
lr_model = joblib.load("lr_model.pkl")
scaler = joblib.load("scaler.pkl")

# Streamlit app
st.title("Student Performance Prediction App")
st.write("This app predicts a student's **Performance Index** based on study habits and lifestyle factors.")

# User inputs
hours = st.number_input("Hours Studied", min_value=0, max_value=24, value=5)
previous_scores = st.number_input("Previous Scores", min_value=0, max_value=100, value=60)
extracurricular = st.selectbox("Extracurricular Activities (0 = No, 1 = Yes)", [0, 1])
sleep = st.number_input("Sleep Hours", min_value=0, max_value=24, value=7)
papers = st.number_input("Sample Question Papers Practiced", min_value=0, max_value=20, value=3)

# Create dataframe for input
new_data = pd.DataFrame({
    'Hours Studied': [hours],
    'Previous Scores': [previous_scores],
    'Extracurricular Activities': [extracurricular],
    'Sleep Hours': [sleep],
    'Sample Question Papers Practiced': [papers]
})

# Scale features
new_data_scaled = scaler.transform(new_data)

# Predict
if st.button("Predict Performance Index"):
    prediction = lr_model.predict(new_data_scaled)[0]
    st.success(f"Predicted Performance Index: **{prediction:.2f}**")
