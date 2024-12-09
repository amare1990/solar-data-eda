import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from windrose import WindroseAxes
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

# Define the time-series plotting function
def plot_time_series(df, time_col, value_cols, title="Time Series Analysis"):
    df[time_col] = pd.to_datetime(df[time_col])
    fig, ax = plt.subplots(figsize=(10, 6))
    df.set_index(time_col)[value_cols].plot(ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Values")
    plt.legend(loc="best")
    return fig

def plot_correlation(df, columns, title):
  corr = df[columns].corr()
  fig, ax = plt.subplots(figsize=(8, 6))
  sns.heatmap(corr, annot=True, cmap='coolwarm', ax=ax)
  ax.set_title(title)
  return fig

def plot_scatter_matrix(df, columns, title):
  fig, ax = plt.subplots(figsize=(12, 10))
  sns.pairplot(df[columns])
  st.pyplot(fig)
  ax.set_title(title)

def plot_wind_rose(df, speed_col, direction_col):
    fig = plt.figure(figsize=(8, 8))
    ax = WindroseAxes.from_ax(fig=fig)
    ax.bar(df[direction_col], df[speed_col], normed=True, opening=0.8, edgecolor="white")
    ax.set_legend()
    return fig

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

  # Time-series analysis
  if st.sidebar.checkbox("Perform Time-Series Analysis"):
    time_col = st.selectbox("Select the Time Column", df.columns)
    value_cols = st.multiselect("Select Value Columns (GHI, DNI, DHI, Tamb) for Analysis", df.columns)

    if time_col and value_cols:
        st.write("### Time-Series Plot")

        fig = plot_time_series(df, time_col, value_cols)
        st.pyplot(fig)
    else:
        st.warning("Please select both a time column and at least one value column.")

  # Checkbox for temperature analysis
  if st.sidebar.checkbox("Temperature Analysis"):
    st.write("## Temperature Analysis")

    temp_columns = [col for col in ['TModA', 'TModB'] if col in df.columns]
    rh_column = 'RH' if 'RH' in df.columns else None
    solar_columns = ['GHI', 'DNI', 'DHI', 'ModA', 'ModB']

    if temp_columns and rh_column:
        st.write("### Correlation Between RH, Solar Radiation, and Temperature")
        analysis_columns = solar_columns + temp_columns + [rh_column]
        analysis_columns = [col for col in analysis_columns if col in df.columns]
        fig = plot_correlation(df, analysis_columns, "RH, Solar Radiation, and Temperature Correlation")
        st.pyplot(fig)
    else:
        missing = []
        if not temp_columns:
            missing.append("Temperature data (TModA, TModB)")
        if not solar_columns:
           missing.append("Solar radiation (GHI, DNI, DHI)")
        if not rh_column:
            missing.append("Relative Humidity (RH)")
        st.warning(f"{', '.join(missing)} not found in the dataset.")

  # Checkbox for wind analysis
  if st.sidebar.checkbox("Wind Analysis"):
    st.write("## Wind Analysis")

    wind_columns = [col for col in ['WS', 'WSgust', 'WD'] if col in df.columns]
    if 'WS' in df.columns and 'WD' in df.columns:
        st.write("### Distribution of Wind Speed and Direction")
        fig = plot_wind_rose(df, speed_col='WS', direction_col='WD')
        st.pyplot(fig)
        st.write("""
        - The wind rose visualizes how often the wind comes from various directions and the distribution of wind speeds.
        """)
    else:
        missing = []
        if 'WS' not in df.columns:
            missing.append("Wind Speed (WS)")
        if 'WD' not in df.columns:
            missing.append("Wind Direction (WD)")
        st.warning(f"{', '.join(missing)} not found in the dataset.")


else:
   st.warning("Please upload a dataset to begin.")

