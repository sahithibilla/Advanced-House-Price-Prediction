import streamlit as st
import pandas as pd
import pickle

st.title("🏡 House Price Prediction App")
st.write("An interactive ML model to estimate house prices.")

# Sidebar inputs
st.sidebar.header("Input House Features")

income = st.sidebar.slider("Avg. Area Income", 0, 200000, 50000)
age = st.sidebar.slider("Avg. Area House Age", 0, 100, 20)
rooms = st.sidebar.slider("Avg. Area Number of Rooms", 1, 15, 5)
bedrooms = st.sidebar.slider("Avg. Area Number of Bedrooms", 1, 10, 3)
population = st.sidebar.slider("Area Population", 0, 1000000, 50000)

# Load model
with open("lr.pkl", "rb") as f:
    model = pickle.load(f)

# ✅ Prepare input in EXACT order used during training
input_data = pd.DataFrame({
    "Avg. Area Income": [income],
    "Avg. Area House Age": [age],
    "Avg. Area Number of Rooms": [rooms],
    "Avg. Area Number of Bedrooms": [bedrooms],
    "Area Population": [population]
})

# Prediction
prediction = model.predict(input_data)[0]

st.subheader("Predicted House Price")
st.success(f"₹ {prediction:,.2f}")
