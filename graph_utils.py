import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

def show_frequency(df: pd.DataFrame, column_name: str) -> None:
    value_counts = df[column_name].value_counts()
    value_counts.plot.bar()
    plt.title(f'Frequency of Values in {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.show()

def show_numerical_attribute_distribution(df: pd.DataFrame) -> None:
    numerical_attributes = df.select_dtypes(include=np.number).columns
    for attribute in numerical_attributes:
        percentile_values = np.percentile(df[attribute], np.arange(0, 101, 10))

        plt.figure(figsize=(8, 6))
        plt.plot(np.arange(0, 101, 10), percentile_values, marker='o')
        plt.title(f'Percentile Distribution for {attribute}')
        plt.xlabel('Percentile')
        plt.ylabel(attribute)
        plt.grid(True)
        plt.show()

def show_categorical_attribute_histogram(df: pd.DataFrame) -> None:
    categorical_attributes = df.select_dtypes(include=object).columns
    for attribute in categorical_attributes:
        plt.figure(figsize=(8, 6))
        df[attribute].value_counts().plot.bar()
        plt.title(f'Frequency of Values in {attribute}')
        plt.xlabel(attribute)
        plt.ylabel('Frequency')
        plt.show()