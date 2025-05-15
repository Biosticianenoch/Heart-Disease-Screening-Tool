
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

# Model training
X_train, X_test, Y_train, Y_test = train_test_split(X, y, test_size=0.2, random_state=42)

xgb = XGBClassifier(
    learning_rate=0.01, 
    n_estimators=25, 
    max_depth=15,
    gamma=0.6, 
    subsample=0.52,
    colsample_bytree=0.6,
    seed=27,
    reg_lambda=2, 
    booster='dart',
    colsample_bylevel=0.6, 
    colsample_bynode=0.5
)

xgb.fit(X_train, Y_train)
xgb_score = xgb.score(X_test, Y_test)
xgb_Y_pred = xgb.predict(X_test)


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


# Preprocessing user input
def preprocess(age, sex, cp, trestbps, restecg, chol, fbs, thalach, exang, oldpeak, slope, ca, thal):
    # Encode categorical variables
    sex = 1 if sex == "male" else 0

    cp_map = {
        "Typical angina": 0,
        "Atypical angina": 1,
        "Non-anginal pain": 2,
        "Asymptomatic": 3
    }
    cp = cp_map[cp]

    restecg_map = {
        "Nothing to note": 0,
        "ST-T Wave abnormality": 1,
        "Possible or definite left ventricular hypertrophy": 2
    }
    restecg = restecg_map[restecg]

    fbs = 1 if fbs == "Yes" else 0
    exang = 1 if exang == "Yes" else 0

    slope_map = {
        "Upsloping: better heart rate with exercise (uncommon)": 0,
        "Flatsloping: minimal change (typical healthy heart)": 1,
        "Downsloping: signs of unhealthy heart": 2
    }
    slope = slope_map[slope]

    # No description in original dataset, so use selected value directly
    thal = int(thal)

    # Construct input vector
    user_input = [age, sex, cp, trestbps, chol, fbs, restecg,
                  thalach, exang, oldpeak, slope, ca, thal]
    
    user_input = np.array(user_input).reshape(1, -1)

    # Scale input
    scaler = StandardScaler()
    scaler.fit(X)  # fit on training data structure
    user_input = scaler.transform(user_input)

    # Make prediction
    prediction = xgb.predict(user_input)
    return prediction

# Run prediction
if st.button("Predict"):
    pred = preprocess(age, sex, cp, trestbps, restecg, chol, fbs,
                      thalach, exang, oldpeak, slope, ca, thal)
    
    if pred[0] == 0:
        st.error('⚠️ Warning! You have a high risk of heart disease.')
    else:
        st.success('✅ You have a lower risk of heart disease.')

# Feedback
feedback = st.sidebar.slider('How much would you rate this app?', min_value=0, max_value=5, step=1)

if feedback:
    st.header("Thank you for rating the app!")
    st.info("⚠️ Caution: This is just a prediction, not medical advice. See a doctor if symptoms persist.")



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

