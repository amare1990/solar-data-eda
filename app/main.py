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

if uploaded_dataset:
  df = pd.read_csv(uploaded_dataset)
  st.write("### Below are the first five samples of the dataset uploaded")
  st.dataframe(df.head())

  st.write("### Key Insights:")
  st.write(f"- **Total Rows**: {df.shape[0]}")
  st.write(f"- **Total Columns**: {df.shape[1]}")
  st.write(f"- **Missing Values**: {df.isnull().sum().sum()}")
  st.write(f"- **Missing Values excluding the Comments column**: {df.drop(columns=['Comments']).isnull().sum().sum()}")



  # Interactive Feature: Correlation Heatmap
  if st.sidebar.checkbox("Correlation analysis"):
    st.write("## Interactive Correlation Heatmap")
    selected_columns = st.multiselect("Select columns for correlation heatmap", df.columns.tolist())
    if selected_columns:
       fig = plot_correlation(df, selected_columns, f"Correlation Heatmap among {selected_columns}")
       st.pyplot(fig)
    else:
       st.warning("Select at least two columns for correlation analysis.")

else:
   st.warning("Please upload a dataset to begin.")

