import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np


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
