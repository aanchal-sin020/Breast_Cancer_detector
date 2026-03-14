import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression

# Load dataset
df = pd.read_csv("breast_cancer.csv")

X = df.iloc[:,1:-1]
y = df.iloc[:,-1]

# Train model
model = LogisticRegression(max_iter=5000)
model.fit(X,y)

st.title("Breast Cancer Detection System")

st.write("Predict whether a tumor is Benign or Malignant")

st.sidebar.header("Input Features")

input_data = []

for feature in X.columns:
    val = st.sidebar.slider(feature, float(X[feature].min()), float(X[feature].max()))
    input_data.append(val)

input_array = np.array(input_data).reshape(1,-1)

if st.button("Predict"):
    
    prediction = model.predict(input_array)

    if prediction[0] == 1:
        st.error("Malignant Tumor Detected")
    else:
        st.success("Benign Tumor")
