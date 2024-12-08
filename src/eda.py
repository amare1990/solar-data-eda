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


def evaluate_cleaning_impact(df, sensor_columns=['ModA', 'ModB'], cleaning_column='Cleaning', time_column='Timestamp'):

    if not all(col in df.columns for col in sensor_columns + [cleaning_column, time_column]):
        raise ValueError("Required columns are missing in the DataFrame.")

    # Separate data into cleaned and uncleaned subsets
    uncleaned_data = df[df[cleaning_column] == 0]
    cleaned_data = df[df[cleaning_column] == 1]

    # Compute summary statistics
    summary_statistics = {
        'uncleaned': uncleaned_data[sensor_columns].describe(),
        'cleaned': cleaned_data[sensor_columns].describe()
    }

    # Plot sensor readings over time, grouped by cleaning status
    plt.figure(figsize=(14, 8))
    for sensor in sensor_columns:
        plt.plot(df[time_column], df[sensor], label=f'{sensor} (Original)', alpha=0.5)
        plt.scatter(
            cleaned_data[time_column], cleaned_data[sensor],
            label=f'{sensor} (Cleaned)', color='red', s=10
        )

    plt.title('Impact of Cleaning on Sensor Readings Over Time')
    plt.xlabel('Time')
    plt.ylabel('Sensor Reading')
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

    return summary_statistics


def visualize_correlations(df, output_dir=None):

    variables = ['GHI', 'DNI', 'DHI', 'TModA', 'TModB', 'WS', 'WSgust', 'WD']
    selected_data = df[variables].dropna()

    # 1. Correlation Matrix
    corr_matrix = selected_data.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
    plt.title("Correlation Matrix")

    if output_dir:
        plt.savefig(f"{output_dir}/correlation_matrix.png")
    else:
        plt.show()

    # 2. Pair Plot for Solar Radiation and Temperature
    sns.pairplot(selected_data[['GHI', 'DNI', 'DHI', 'TModA', 'TModB']])
    plt.suptitle("Pair Plot: Solar Radiation and Temperature", y=1.02)

    if output_dir:
        plt.savefig(f"{output_dir}/pair_plot_solar_temp.png")
    else:
        plt.show()

    sns.pairplot(selected_data[['GHI', 'DNI', 'DHI', 'WS', 'WSgust', 'WD']], kind='scatter')
    plt.suptitle("Scatter Matrix: Wind Conditions and Solar Irradiance", y=1.02)

    if output_dir:
        plt.savefig(f"{output_dir}/scatter_matrix_wind_solar.png")
    else:
        plt.show()

    return corr_matrix

# Wind Analysis
from windrose import WindroseAxes

def wind_analysis(df, output_dir=None):

    # Check if required columns exist
    if not {'WS', 'WD'}.issubset(df.columns):
        raise ValueError("Dataset must contain 'WS' (Wind Speed) and 'WD' (Wind Direction) columns.")

    # Drop rows with missing values in wind columns
    wind_data = df[['WS', 'WD']].dropna()

    # Create a wind rose plot
    plt.figure(figsize=(8, 8))
    ax = WindroseAxes.from_ax()
    ax.bar(
        wind_data['WD'],
        wind_data['WS'],
        normed=True,  # Normalize the frequency
        opening=0.8,
        edgecolor='white'
    )
    ax.set_legend(title="Wind Speed (m/s)", loc="lower right", bbox_to_anchor=(1.2, 0.1))
    plt.title("Wind Rose: Distribution of Wind Speed and Direction")

    # Save or display the plot
    if output_dir:
        plt.savefig(f"{output_dir}/wind_rose.png", bbox_inches='tight')
    else:
        plt.show()

    # Variability analysis: Boxplot for wind speed
    plt.figure(figsize=(8, 5))
    wind_data['WS'].plot.box()
    plt.title("Wind Speed Variability")
    plt.ylabel("Wind Speed (m/s)")
    plt.grid(True)

    # Save or display the boxplot
    if output_dir:
        plt.savefig(f"{output_dir}/wind_speed_variability.png", bbox_inches='tight')
    else:
        plt.show()


# Effect of humudity over Temprature and solar radiation
def temperature_analysis(df, output_dir=None):

    # Check if required columns exist
    required_columns = {'RH', 'TModA', 'TModB', 'GHI', 'DNI', 'DHI'}
    if not required_columns.issubset(df.columns):
        raise ValueError(f"Dataset must contain the following columns: {required_columns}")

    # Drop rows with missing values in relevant columns
    filtered_df = df[list(required_columns)].dropna()

    # 1. Scatter plot: RH vs Temperature (TModA)
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=filtered_df['RH'], y=filtered_df['TModA'], alpha=0.7)
    plt.title("Relative Humidity vs Temperature (TModA)")
    plt.xlabel("Relative Humidity (%)")
    plt.ylabel("Temperature (°C)")
    plt.grid(True)

    # Save or display the plot
    if output_dir:
        plt.savefig(f"{output_dir}/rh_vs_temperature_modA.png", bbox_inches='tight')
    else:
        plt.show()

    # 2. Scatter plot: RH vs Solar Radiation (GHI)
    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=filtered_df['RH'], y=filtered_df['GHI'], alpha=0.7, color='orange')
    plt.title("Relative Humidity vs Solar Radiation (GHI)")
    plt.xlabel("Relative Humidity (%)")
    plt.ylabel("Global Horizontal Irradiance (W/m²)")
    plt.grid(True)

    # Save or display the plot
    if output_dir:
        plt.savefig(f"{output_dir}/rh_vs_solar_radiation_ghi.png", bbox_inches='tight')
    else:
        plt.show()

    # 3. Correlation Matrix
    correlation_matrix = filtered_df.corr()
    plt.figure(figsize=(10, 8))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        cbar=True
    )
    plt.title("Correlation Matrix: Temperature, RH, and Solar Radiation")

    # Save or display the heatmap
    if output_dir:
        plt.savefig(f"{output_dir}/correlation_matrix.png", bbox_inches='tight')
    else:
        plt.show()


# Create Histograms

def create_histograms(df, columns, output_dir=None):

    # Filter only the columns that exist in the DataFrame
    available_columns = [col for col in columns if col in df.columns]
    if not available_columns:
        raise ValueError("None of the specified columns are available in the dataset.")

    # Drop missing values to avoid skewed distributions
    filtered_df = df[available_columns].dropna()

    # Create histograms
    for col in available_columns:
        plt.figure(figsize=(8, 5))
        sns.histplot(filtered_df[col], kde=True, bins=30, color='blue', alpha=0.7)
        plt.title(f"Histogram of {col}")
        plt.xlabel(col)
        plt.ylabel("Frequency")
        plt.grid(True)

        # Save or display the plot
        if output_dir:
            plt.savefig(f"{output_dir}/histogram_{col}.png", bbox_inches='tight')
        else:
            plt.show()

