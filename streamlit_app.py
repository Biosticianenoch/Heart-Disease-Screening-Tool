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


