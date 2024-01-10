import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import scipy

def show_frequency(test: pd.Series, column_name: str, index: str) -> None:
    value_counts = test.value_counts()
    value_counts.plot.bar()
    plt.title(f'Frequency of Values in {column_name}. Split: {index}')
    plt.xlabel(column_name)
    plt.ylabel('Frequency')
    plt.show()

def show_numerical_attribute_distribution(numerical_attributes: pd.Series, index: str) -> None:
    for attribute in numerical_attributes.columns:
        counts = []
        min_value = numerical_attributes[attribute].min()
        max_value = numerical_attributes[attribute].max()
        for p in range (0, 10):
            percentile = min_value + (max_value - min_value) * p / 10
            next_percentile = min_value + (max_value - min_value) * (p + 1) / 10
            count = numerical_attributes[(numerical_attributes[attribute] >= percentile) & (numerical_attributes[attribute] <= next_percentile)][attribute].count()
            counts.append(count)
        plt.figure(figsize=(6, 4))
        x = np.arange(0.5, 10.5, 1)
        y = np.array(counts)
        plt.bar(x, y)
        plt.title(f'Distribution of {attribute}. Split: {index}')
        plt.xlabel('Percentile')
        plt.ylabel('Frequency')
        plt.xticks(ticks = range(0, 10), labels = range(0, 100, 10))
        plt.show()

def show_categorical_attribute_histogram(categorical_attributes: pd.Series, index: str) -> None:
    for attribute in categorical_attributes.columns:
        plt.figure(figsize=(6, 4))
        categorical_attributes[attribute].value_counts().plot.bar()
        plt.title(f'Frequency of Values in {attribute}. Split: {index}')
        plt.xlabel(attribute)
        plt.ylabel('Frequency')
        plt.xticks(rotation=45, ha='right')
        plt.show()

def show_numerical_corellation(numerical_attributes: pd.Series, test: pd.Series, index: str) -> None:
    correlation_results = pd.DataFrame(columns=['Coefficient', 'P-value'])
    for attribute in numerical_attributes.columns:
        coef, p_value = scipy.stats.pointbiserialr(numerical_attributes[attribute], test)
        correlation_results.loc[attribute] = [coef, p_value]

    print(correlation_results)

    significant_attributes = correlation_results[correlation_results['P-value'] <= 0.05]
    plt.figure(figsize=(10, 4))
    plt.bar(significant_attributes.index, significant_attributes['Coefficient'])
    plt.title(f'Point-Biserial Correlation with Revenue. Split: {index}')
    plt.xlabel('Attribute')
    plt.ylabel('Point-Biserial Correlation Coefficient')
    plt.xticks(rotation=45, ha='right')
    plt.show()

def show_categorical_corellation(categorical_attributes: pd.Series, test: pd.Series, index: str) -> None:
    chi2_results = pd.DataFrame(columns=['Chi2', 'P-value'])

    for attribute in categorical_attributes.columns:
        contingency_table = pd.crosstab(categorical_attributes[attribute], test)
        chi2, p_value, _, _ = scipy.stats.chi2_contingency(contingency_table)
        chi2_results.loc[attribute] = [chi2, p_value]

    print(chi2_results)

    significant_attributes = chi2_results[chi2_results['P-value'] <= 0.05]
    plt.figure(figsize=(10, 4))
    plt.bar(significant_attributes.index, significant_attributes['Chi2'])
    plt.title(f'Chi-squared Statistic for Categorical Attributes with Revenue. Split: {index}')
    plt.xlabel('Attribute')
    plt.ylabel('Chi-squared Statistic')
    plt.xticks(rotation=45, ha='right')
    plt.show()