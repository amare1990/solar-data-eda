import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Streamlit CSS styling
st.markdown(
  """
<style>
.title{
font-size: 27px;
        font-weight: bold;
        text-align: center;
        color: #4CAF50; /* Moonlight Green */
        margin-bottom: 20px;
}
.welcome{
}
</style>
""", unsafe_allow_html=True
)

# Helper function: Plot correlation heatmap

def plot_correlation(df, columns, title):
  corr = df[columns].corr()
  fig, ax = plt.subplots(figsize=(8, 6))
  sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
  ax.set_title(title)
  return fig

def plot_scatter_matrix(df, columns):
  fig, ax = plt.subplots(figsize=(12, 10))
  sns.pairplot(df[columns])
  st.pyplot(fig)

st.markdown('<div class="title">Solar Energy Statistical and Exploratory Data Analysis Dashboard</div', unsafe_allow_html=True)
st.write('Welcome to the MoonLight Energy Solutions dashboard')
st.write("Explore solar energy data insights for Benin, Togo, and Sierra Leone datasets.")

# File upload section
st.sidebar.header("Upload a dataset here")
uploaded_dataset = st.sidebar.file_uploader("Upload a CSV file", type=["CSV"])

