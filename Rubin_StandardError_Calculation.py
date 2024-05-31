"""
The purpose of this code is to calculate the Standard Error (SE) of a set of imputed datasets using Rubin's formula. Steps Involved:
Reading the CSV File:
The code reads a CSV file containing multiple columns, each representing an imputed dataset. This is done using the pandas library.
Extracting Imputed Datasets:
Each column of the CSV file is extracted as a separate numpy array, which represents one imputed dataset. Missing values (if any) are removed using the dropna() method.
Computing Within-Imputation Variance:
For each imputed dataset, the within-imputation variance is calculated. This variance represents the variability within each individual dataset.
Computing Between-Imputation Variance:
The mean of each imputed dataset is computed first. Then, the variance of these means is calculated, representing the variability between the different imputed datasets.
Combining Variances Using Rubin's Formula:
Rubin's formula is used to combine the within-imputation variance and the between-imputation variance to obtain the total variance. This accounts for both the within-dataset variability and the between-dataset variability.
Calculating the Standard Error:
The standard error is computed as the square root of the total variance. This standard error quantifies the overall uncertainty in the imputed datasets.
Usage:
To use this code, you need to have a CSV file where each column contains an imputed dataset. By running the code, you can read the datasets from the CSV file and calculate the standard error using Rubin's formula. 
"""
import pandas as pd
import numpy as np

# Function to calculate within-imputation variance
def within_imputation_variance(imputed_datasets):
    within_vars = []
    for dataset in imputed_datasets:
        var = np.var(dataset, ddof=1)  # Using ddof=1 to get sample variance
        within_vars.append(var)
    return np.mean(within_vars)

# Function to calculate between-imputation variance
def between_imputation_variance(imputed_datasets):
    means = [np.mean(dataset) for dataset in imputed_datasets]
    mean_of_means = np.mean(means)
    between_var = np.var(means, ddof=1)  # Using ddof=1 to get sample variance
    return between_var

# Function to compute total variance using Rubin's formula
def rubins_total_variance(imputed_datasets):
    m = len(imputed_datasets)
    W = within_imputation_variance(imputed_datasets)
    B = between_imputation_variance(imputed_datasets)
    total_var = W + (1 + 1/m) * B
    return total_var

# Function to compute standard error using Rubin's formula
def standard_error(imputed_datasets):
    total_var = rubins_total_variance(imputed_datasets)
    return np.sqrt(total_var)

# Read the CSV file containing imputed datasets
csv_file_path = 'imputed_datasets.csv'
data = pd.read_csv(csv_file_path)

# Assuming each column represents an imputed dataset
imputed_datasets = [data[col].dropna().values for col in data.columns]

# Calculate the standard error
se = standard_error(imputed_datasets)
print(f'Standard Error: {se}')
