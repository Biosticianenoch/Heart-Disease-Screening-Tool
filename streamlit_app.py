import streamlit as st
import pandas as pd


st.title('🎈 Heart Disease Screening Tool')

st.write('This Tool is used to assess the heart disease status of individuals using machine learning integration!')

with st.expander("Data"):
    st.write("Raw data")
    df = pd.read_csv('https://raw.githubusercontent.com/Biosticianenoch/data/refs/heads/main/heart.csv')
    df

st.write("**X**")
X = df.drop("target", axis=1)
st.write(X)

st.write("**y**")
y = df["target"]
st.write(y)

import altair as alt

with st.expander("Data Visualization"):
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X('target:N', title='Heart Disease'),
        y=alt.Y('count():Q', title='Count')
    ).properties(title='Heart Disease Distribution')

    st.altair_chart(chart, use_container_width=True)

with st.sidebar:
    st.header("Input Features")
    # User input fields
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


    # front end elements of the web page
html_temp = """
    <div style ="background-color:pink;padding:13px">
    <h1 style ="color:black;text-align:center;">Heart Disease Prediction Dashboard</h1>
    </div>
    """

# display the front end aspect
st.markdown(html_temp, unsafe_allow_html = True)
st.subheader('by Enock Bereka')

st.sidebar.subheader("About App")

st.sidebar.info("This web app is helps you to find out whether you are at a risk of developing a heart disease.")
st.sidebar.info("Enter the required fields and click on the 'Predict' button to check whether you have a healthy heart")
st.sidebar.info("Don't forget to rate this app")



feedback = st.sidebar.slider('How much would you rate this app?',min_value=0,max_value=5,step=1)

if feedback:
  st.header("Thank you for rating the app!")
  st.info("Caution: This is just a prediction and not doctoral advice. Kindly see a doctor if you feel the symptoms persist.")
