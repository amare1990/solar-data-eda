import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def load_datasets():
    benin_df = pd.read_csv("../datasets/benin-malanville.csv")
    sierra_df = pd.read_csv("../datasets/sierraleone-bumbuna.csv")
    togo_df = pd.read_csv("../datasets/togo-dapaong_qc.csv")
    return benin_df, sierra_df, togo_df


def summary_statistics(df):
    return df.describe()

def check_data_quality(df):
    """
    Perform data quality checks on the dataset.
    Includes checks for missing values, outliers, and incorrect entries.

    Args:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    dict: A dictionary containing results of the data quality checks.
    """
    # Initialize the result dictionary
    quality_report = {}

    # 1. Check for missing values
    # missing_values = df.isnull().sum()
    missing_values = df.drop(columns=['Comments'], errors='ignore').isnull().sum()
    quality_report['missing_values'] = missing_values[missing_values > 0]



    # 2. Check for negative values in GHI, DNI, DHI (if they should always be positive)
    invalid_values = {}
    for col in ['GHI', 'DNI', 'DHI']:
        if col in df.columns:

            invalid_values[col] = df[df[col] < 0][col].count()
    quality_report['negative_values'] = invalid_values

    # 3. Check for outliers in sensor readings (ModA, ModB) and wind data (WS, WSgust)
    # Using the IQR (Interquartile Range) method for outlier detection
    outliers = {}
    for col in ['ModA', 'ModB', 'WS', 'WSgust']:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col].count()
    quality_report['outliers'] = outliers

    # 4. Add a summary of total issues detected
    total_issues = {
        'total_missing': sum(quality_report['missing_values']),
        'total_negative': sum(quality_report['negative_values'].values()),
        'total_outliers': sum(quality_report['outliers'].values())
    }
    quality_report['summary'] = total_issues

    return quality_report


def plot_time_series(df, time_col, value_cols, title="Time Series Analysis"):
    df[time_col] = pd.to_datetime(df[time_col])
    df.set_index(time_col)[value_cols].plot(figsize=(10, 6))
    plt.title(title)
    plt.show()
