import streamlit as st
import pandas as pd
import numpy as np
import joblib as jb

st.title("🔆 Solar Power Output Prediction App")
st.write("Enter environmental parameters to predict Solar Power Output")

# Load trained multiple regression model
model = jb.load(r"C:\Users\student\Downloads\solar_power_prediction_model.pkl")

# ---- User Inputs ----
temp = st.number_input("Enter Temperature (°C):", 
                       min_value=0.0, max_value=60.0, value=25.0)

humidity = st.number_input("Enter Humidity (%):", 
                           min_value=0.0, max_value=100.0, value=50.0)

irradiance = st.number_input("Enter Solar Irradiance (W/m²):", 
                             min_value=0.0, max_value=1500.0, value=800.0)

wind_speed = st.number_input("Enter Wind Speed (m/s):", 
                             min_value=0.0, max_value=50.0, value=5.0)

# ---- Prediction Button ----
if st.button("Predict Solar Power Output"):
    
    # Arrange inputs in same order used during training
    new_data = np.array([[temp, humidity, irradiance, wind_speed]])
    
    prediction = model.predict(new_data)
    
    st.success(f"Predicted Solar Power Output: {prediction[0]:.2f} kW")

st.write("Thank You! 🌞")
