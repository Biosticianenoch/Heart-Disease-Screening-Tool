
import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# Title and description
st.title('🎈 Heart Disease Screening Tool')
st.write('This tool assesses heart disease risk using machine learning integration!')

# Load and display data
with st.expander("Data"):
    st.write("Raw data")
    df = pd.read_csv('https://raw.githubusercontent.com/Biosticianenoch/data/refs/heads/main/heart.csv')
    st.dataframe(df)

# Features and target
st.write("**X**")
X = df.drop("target", axis=1)
st.write(X)

st.write("**y**")
y = df["target"]
st.write(y)

# Visualization
with st.expander("Data Visualization"):
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('target:N', title='Heart Disease'),
        y=alt.Y('count():Q', title='Count')
    ).properties(title='Heart Disease Distribution')
    st.altair_chart(chart, use_container_width=True)

# Sidebar user input
with st.sidebar:
    st.header("Input Features")
    age = st.selectbox("Age", range(1, 121))
    sex = st.radio("Select Gender:", ('male', 'female'))
    cp = st.selectbox('Chest Pain Type', (
        "Typical angina", 
        "Atypical angina", 
        "Non-anginal pain", 
        "Asymptomatic"
    ))
    trestbps = st.selectbox('Resting Blood Pressure (mm Hg)', range(1, 500))
    restecg = st.selectbox('Resting Electrocardiographic Results', (
        "Nothing to note", 
        "ST-T Wave abnormality", 
        "Possible or definite left ventricular hypertrophy"
    ))
    chol = st.selectbox('Serum Cholesterol in mg/dl', range(1, 1000))
    fbs = st.radio("Fasting Blood Sugar > 120 mg/dl", ['Yes', 'No'])
    thalach = st.selectbox('Maximum Heart Rate Achieved', range(1, 300))
    exang = st.selectbox('Exercise Induced Angina', ["Yes", "No"])
    oldpeak = st.number_input('Oldpeak (ST depression induced by exercise)', value=0.0)
    slope = st.selectbox('Heart Rate Slope', (
        "Upsloping: better heart rate with exercise (uncommon)",
        "Flatsloping: minimal change (typical healthy heart)",
        "Downsloping: signs of unhealthy heart"
    ))
    ca = st.selectbox('Number of Major Vessels Colored by Fluoroscopy', range(0, 5))
    thal = st.selectbox('Thalium Stress Result (1 = normal; 2 = fixed defect; 3 = reversible defect)', range(1, 8))

# HTML header
html_temp = """
    <div style ="background-color:pink;padding:13px">
    <h1 style ="color:black;text-align:center;">Heart Disease Prediction Dashboard</h1>
    </div>
    """
st.markdown(html_temp, unsafe_allow_html=True)
st.subheader('by Enock Bereka')

# Sidebar Info
st.sidebar.subheader("About App")
st.sidebar.info("This web app helps assess your risk for heart disease.")
st.sidebar.info("Enter the required fields and click 'Predict' to check your risk.")
st.sidebar.info("Don't forget to rate this app.")

