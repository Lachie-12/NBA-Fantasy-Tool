import streamlit as st
import pandas as pd

st.title("NBA Fantasy Rankings")

df = pd.read_csv("zscore_rankings_2025-26.csv")
st.dataframe(df)