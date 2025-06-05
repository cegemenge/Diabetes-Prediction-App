
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load dataset
df = pd.read_csv("diabetes.csv")
X = df.drop("Outcome", axis=1)
y = df["Outcome"]

# Train model
model = RandomForestClassifier()
model.fit(X, y)

st.title("Diabetes Prediction App")
st.markdown("Enter your health details to predict if you are likely diabetic (1) or not (0)")

# Create sliders for input
user_input = []
for col in X.columns:
    val = st.slider(col, float(X[col].min()), float(X[col].max()), float(X[col].mean()))
    user_input.append(val)

# Make prediction
if st.button("Predict"):
    prediction = model.predict([user_input])
    st.success(f"Prediction: {'Diabetic (1)' if prediction[0] == 1 else 'Not Diabetic (0)'}")
