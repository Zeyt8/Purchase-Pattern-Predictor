import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy

def show_frequency(df: pd.DataFrame, column_name: str) -> None:
    value_counts = df[column_name].value_counts()
    value_counts.plot.bar()
    plt.title(f'Frequency of Values in {column_name}')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.show()

def show_numerical_attribute_distribution(df: pd.DataFrame) -> None:
    numerical_attributes = df.columns
    for attribute in numerical_attributes:
        counts = []
        min_value = df[attribute].min()
        max_value = df[attribute].max()
        for p in range (0, 10):
            percentile = min_value + (max_value - min_value) * p / 10
            next_percentile = min_value + (max_value - min_value) * (p + 1) / 10
            # how many values are less than or equal to the percentile
            count = df[(df[attribute] >= percentile) & (df[attribute] <= next_percentile)][attribute].count()
            counts.append(count)
        plt.figure(figsize=(6, 4))
        x = np.arange(0.5, 10.5, 1)
        y = np.array(counts)
        plt.bar(x, y)
        plt.title(f'Distribution of {attribute}')
        plt.xlabel('Percentile')
        plt.ylabel('Frequency')
        plt.xticks(ticks = range(0, 10), labels = range(0, 100, 10))
        plt.show()

def show_categorical_attribute_histogram(df: pd.DataFrame) -> None:
    categorical_attributes = df.columns
    for attribute in categorical_attributes:
        plt.figure(figsize=(6, 4))
        df[attribute].value_counts().plot.bar()
        plt.title(f'Frequency of Values in {attribute}')
        plt.xlabel(attribute)
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, ha='right')
        plt.show()

def show_numerical_corellation(df: pd.DataFrame) -> None:
    numerical_attributes = df.columns
    correlation_results = pd.DataFrame(columns=['Coefficient', 'P-value'])
    for attribute in numerical_attributes:
        coef, p_value = scipy.stats.pointbiserialr(df[attribute], df['Revenue'])
        correlation_results.loc[attribute] = [coef, p_value]

    print(correlation_results)

    significant_attributes = correlation_results[correlation_results['P-value'] <= 0.05]
    plt.figure(figsize=(10, 4))
    plt.bar(significant_attributes.index, significant_attributes['Coefficient'])
    plt.title('Point-Biserial Correlation with Revenue')
    plt.xlabel('Attribute')
    plt.ylabel('Point-Biserial Correlation Coefficient')
    plt.xticks(rotation=45, ha='right')
    plt.show()

def show_categorical_corellation(df: pd.DataFrame) -> None:
    categorical_attributes = df.columns
    chi2_results = pd.DataFrame(columns=['Chi2', 'P-value'])

    for attribute in categorical_attributes:
        contingency_table = pd.crosstab(df[attribute], df['Revenue'])
        chi2, p_value, _, _ = scipy.stats.chi2_contingency(contingency_table)
        chi2_results.loc[attribute] = [chi2, p_value]

    print(chi2_results)

    significant_attributes = chi2_results[chi2_results['P-value'] <= 0.05]
    plt.figure(figsize=(10, 4))
    plt.bar(significant_attributes.index, significant_attributes['Chi2'])
    plt.title('Chi-squared Statistic for Categorical Attributes with Revenue')
    plt.xlabel('Attribute')
    plt.ylabel('Chi-squared Statistic')
    plt.xticks(rotation=45, ha='right')
    plt.show()