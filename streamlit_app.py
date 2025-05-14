import streamlit as st
import pandas as pd


st.title('🎈 Heart Disease Screening Tool')

st.write('This Tool is used to assess the heart disease status of individuals using machine learning integration!')

df = pd.read_csv('https://raw.githubusercontent.com/Biosticianenoch/data/refs/heads/main/heart.csv')
df 
