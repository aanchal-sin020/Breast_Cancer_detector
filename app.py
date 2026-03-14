import streamlit as st
import pandas as pd
import numpy as np
import pickle
from sklearn.linear_model import LogisticRegression


st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffe6f0;
    }

.main-title {
    text-align: center;
    font-size: 60px;
    color: #c2185b;
    font-weight: bold;
    margin-bottom: 20px;
}

    .stButton>button {
        background-color: #f06292;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: bold;
    }

    .stButton>button:hover {
        background-color: #ec407a;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Load dataset
df = pd.read_csv("breast_cancer.csv")

X = df.iloc[:,1:-1]
y = df.iloc[:,-1]

# Train model
model = LogisticRegression(max_iter=5000)
model.fit(X,y)

st.markdown(
    '<p class="main-title">🎗️ Breast Cancer Detection System</p>',
    unsafe_allow_html=True
)

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
