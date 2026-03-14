import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression

# --------- PAGE CONFIG ----------
st.set_page_config(page_title="Breast Cancer Detector", layout="wide")

# --------- CUSTOM CSS ----------
st.markdown(
    """
    <style>
    .stApp {
        background-color: #ffe6f0;
    }

    .main-title {
        text-align: center;
        font-size: 70px !important;
        color: #c2185b;
        font-weight: 900;
        margin-bottom: 10px;
    }

    .subtitle {
        text-align: center;
        font-size: 22px;
        color: #444;
        margin-bottom: 30px;
    }

    .stButton>button {
        background-color: #f06292;
        color: white;
        border-radius: 8px;
        border: none;
        font-weight: bold;
        padding: 10px 30px;
        font-size: 18px;
    }

    .stButton>button:hover {
        background-color: #ec407a;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --------- TITLE ----------
st.markdown(
    '<p class="main-title">🎗 Breast Cancer Detection System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="subtitle">Predict whether a tumor is Benign or Malignant using Machine Learning</p>',
    unsafe_allow_html=True
)

# --------- LOAD DATA ----------
df = pd.read_csv("breast_cancer.csv")

X = df.iloc[:,1:-1]
y = df.iloc[:,-1]

# --------- TRAIN MODEL ----------
model = LogisticRegression(max_iter=5000)
model.fit(X,y)

# --------- SIDEBAR INPUT ----------
st.sidebar.header("Input Features")

input_data = []

for feature in X.columns:
    val = st.sidebar.slider(
        feature,
        float(X[feature].min()),
        float(X[feature].max()),
        float(X[feature].mean())
    )
    input_data.append(val)

input_array = np.array(input_data).reshape(1,-1)

# --------- PREDICTION ----------
if st.button("Predict"):

    prediction = model.predict(input_array)

    if prediction[0] == 1:
        st.error("Malignant Tumor Detected")
    else:
        st.success("Benign Tumor")
