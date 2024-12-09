import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from windrose import WindroseAxes
import numpy as np
from scipy import stats
import plotly.express as px

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

# Histograms
def plot_histogram(df, column, bins=30, title="Histogram"):
    fig, ax = plt.subplots(figsize=(8, 6))
    df[column].dropna().hist(bins=bins, ax=ax, color="skyblue", edgecolor="black")
    ax.set_title(title)
    ax.set_xlabel(column)
    ax.set_ylabel("Frequency")
    return fig


# Bubble chart
def plot_bubble_chart(df, x_col, y_col, size_col, color_col=None, title="Bubble Chart"):
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        size=size_col,
        color=color_col,
        hover_name=df.index,
        title=title,
        labels={x_col: x_col, y_col: y_col, size_col: size_col, color_col: color_col},
        template="plotly_white"
    )
    st.plotly_chart(fig, use_container_width=True)



# Data Quality Check
def check_data_quality(df):
    quality_report = {}

    # Check for missing values
    missing_values = df.isnull().sum()
    quality_report['missing_values'] = missing_values[missing_values > 0]

    # Check for negative values in GHI, DNI, DHI
    invalid_values = {}
    for col in ['DHI', 'DNI', 'GHI']:
        if col in df.columns:
            invalid_values[col] = df[df[col] < 0][col].count()
    quality_report['negative_values'] = invalid_values

    # Check for outliers in sensor readings and wind speed data using Z-score
    outlier_report = {}
    outlier_columns = ['ModA', 'ModB', 'WS', 'WSgust']
    for col in outlier_columns:
        if col in df.columns:
            z_scores = np.abs(stats.zscore(df[col].dropna()))
            outlier_report[col] = (z_scores > 3).sum()
    quality_report['outliers'] = outlier_report

    return quality_report

# Clean Data
def clean_data(df):

    # Step 1: Remove Comments column if it exists
    if 'Comments' in df.columns:
        df_cleaned = df.drop(columns=['Comments'])
    else:
        df_cleaned = df.copy()

    # Step 2: Remove rows with negative values in GHI, DNI, DHI
    for col in ['DHI', 'DNI', 'GHI']:
        if col in df_cleaned.columns:
            df_cleaned = df_cleaned[df_cleaned[col] >= 0]

    # Step 3: Remove outliers using Z-scores
    outlier_columns = ['ModA', 'ModB', 'WS', 'WSgust']
    for col in outlier_columns:
        if col in df_cleaned.columns:
            z_scores = np.abs(stats.zscore(df_cleaned[col].dropna()))
            df_cleaned = df_cleaned[(z_scores < 3).fillna(True)]

    # Step 4: Handle missing values by dropping rows with NaNs in critical columns
    critical_columns = ['DHI', 'DNI', 'GHI', 'ModA', 'ModB', 'WS', 'WSgust']
    df_cleaned = df_cleaned.dropna(subset=critical_columns)

    return df_cleaned



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


  # Summary statistics to show mean, media,standard deviation, and the like
  if st.sidebar.checkbox("Summary Statistics"):
          st.title("Summary statistics for datasets")
          st.write("## Summary Statistics")

          numeric_cols = df.select_dtypes(include='number').columns.tolist()
          selected_columns = st.multiselect(
              "Select columns for summary statistics",
              options=numeric_cols,
              default=numeric_cols
          )

          if selected_columns:
              summary = df[selected_columns].describe()
              st.write(summary)
          else:
              st.warning("No numeric columns selected for summary statistics.")

  # Interactive Feature: Correlation Heatmap
  if st.sidebar.checkbox("Correlation analysis"):
    st.title("Correlation analysis for features")
    st.write("## Interactive Correlation Heatmap")
    selected_columns = st.multiselect("Select columns for correlation heatmap", df.columns.tolist())
    if selected_columns:
       fig = plot_correlation(df, selected_columns, f"Correlation Heatmap among {selected_columns}")
       st.pyplot(fig)
    else:
       st.warning("Select at least two columns for correlation analysis.")

  # Time-series analysis
  if st.sidebar.checkbox("Perform Time-Series Analysis"):
    st.title("Perform Time-Series Analysis")
    time_col = st.selectbox("Select the Time Column", df.columns)
    value_cols = st.multiselect("Select Value Columns (GHI, DNI, DHI, Tamb) for Analysis", df.columns)

    if time_col and value_cols:
        st.write("### Time-Series Plot")

        fig = plot_time_series(df, time_col, value_cols)
        st.pyplot(fig)
    else:
        st.warning("Please select both a time column and at least one value column.")


  # Effect of Cleaning
  if st.sidebar.checkbox("Evaluate Cleaning Impact"):
       st.title("Effect of Cleaning feature over ModA and ModB over time")
       st.write("## Cleaning Impact on Sensor Readings")

       required_columns = ['ModA', 'ModB', 'Cleaning', 'Timestamp']
       missing_columns = [col for col in required_columns if col not in df.columns]
       if missing_columns:
           st.warning(f"The following required columns are missing from the dataset: {', '.join(missing_columns)}")
        #    return

       df['Timestamp'] = pd.to_datetime(df['Timestamp'])
       uncleaned_data = df[df['Cleaning'] == 0]
       cleaned_data = df[df['Cleaning'] == 1]
       fig, ax = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
       # Plot ModA
       ax[0].plot(uncleaned_data['Timestamp'], uncleaned_data['ModA'], label='Original ModA', color='red', alpha=0.6)
       ax[0].plot(cleaned_data['Timestamp'], cleaned_data['ModA'], label='Cleaned ModA', color='blue', alpha=0.6)
       ax[0].set_title("Impact on ModA")
       ax[0].set_ylabel("ModA Reading")
       ax[0].legend()

       # Plot ModB
       ax[1].plot(uncleaned_data['Timestamp'], uncleaned_data['ModB'], label='Original ModB', color='red', alpha=0.6)
       ax[1].plot(cleaned_data['Timestamp'], cleaned_data['ModB'], label='Cleaned ModB', color='blue', alpha=0.6)
       ax[1].set_title("Impact on ModB")
       ax[1].set_ylabel("ModB Reading")
       ax[1].legend()

       plt.xlabel("Time")
       plt.tight_layout()

       # Display the plots
       st.pyplot(fig)

       # Provide summary statistics before and after cleaning
       st.write("### Summary Statistics Before and After Cleaning")
       for col in ['ModA', 'ModB']:
           st.write(f"#### {col}")
           original_stats = uncleaned_data[col].describe()
           cleaned_stats = cleaned_data[col].describe()
           st.write("**Original Data**")
           st.write(original_stats)
           st.write("**Cleaned Data**")
           st.write(cleaned_stats)



  # Checkbox for temperature analysis
  if st.sidebar.checkbox("Temperature Analysis"):
    st.title("Temperature Analysis")
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
    st.title("Wind Analysis")
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


  # Checkbox for histograms
  if st.sidebar.checkbox("Histograms"):
      st.title("Histograms")
      st.write("## Histograms")
      histogram_columns = [col for col in ['GHI', 'DNI', 'DHI', 'WS', 'TModA', 'TModB'] if col in df.columns]

      if histogram_columns:
         st.write("### Frequency Distribution of Variables")
         for col in histogram_columns:
             st.write(f"#### {col}")
             fig = plot_histogram(df, col, bins=30, title=f"Histogram of {col}")
             st.pyplot(fig)
      else:
          st.warning("No relevant variables found in the dataset for histogram plotting.")

  # Perform data qualit checking
  if st.sidebar.checkbox("Perform Data Quality Check"):
        st.title("Perform Data Quality Check")
        st.write("## Data Quality Report")
        quality_report = check_data_quality(df)
        st.json(quality_report)

  # Checkbox for Data Cleaning
  if st.sidebar.checkbox("Perform Data Cleaning"):
      st.title("Perform Data Cleaning")
      st.write("## Data Cleaning")
      cleaned_df = clean_data(df)

      # Display Cleaned Data
      st.write("### Cleaned Dataset")
      st.dataframe(cleaned_df)

      # Comparison: Initial vs Cleaned Data
      st.write("### Comparison: Initial vs. Cleaned Data")
      col1, col2 = st.columns(2)
      with col1:
          st.write("#### Initial Dataset")
          st.dataframe(df)
      with col2:
          st.write("#### Cleaned Dataset")
          st.dataframe(cleaned_df)

  # Bubble chart
  # Checkbox for Bubble Chart
  if st.sidebar.checkbox("Explore Relationships with Bubble Charts"):
     st.title("Bubble Chart Analysis Dashboard")
     st.write("## Bubble Chart Analysis")

     # Dropdowns to select columns for the chart
     x_col = st.selectbox("Select X-axis Variable", options=df.columns)
     y_col = st.selectbox("Select Y-axis Variable", options=df.columns)
     size_col = st.selectbox("Select Bubble Size Variable", options=df.columns)
     color_col = st.selectbox("Select Bubble Color Variable (Optional)", options=[None] + list(df.columns), index=0)

     # Ensure the selected columns are numeric
     numeric_columns = df.select_dtypes(include=['number']).columns
     if x_col in numeric_columns and y_col in numeric_columns and size_col in numeric_columns:
        # Plot the bubble chart
        plot_bubble_chart(df, x_col, y_col, size_col, color_col, title="Bubble Chart: Explore Relationships")
     else:
        st.warning("Please select numeric columns for the X-axis, Y-axis, and bubble size.")



else:
   st.warning("Please upload a dataset to begin.")

