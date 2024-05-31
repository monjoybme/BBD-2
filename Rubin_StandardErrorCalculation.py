"""
### Data Summary**
#### CSV File Structure
The CSV file contains the following columns:
1. `dataset_name`: Names of the imputed datasets.
2. `sensitivity_proportion`: Sensitivity proportions for each imputed dataset.
3. `sensitivity_variance`: Sensitivity variances for each imputed dataset.
4. `specificity_proportion`: Specificity proportions for each imputed dataset.
5. `specificity_variance`: Specificity variances for each imputed dataset.
6. `precision_proportion`: Precision proportions for each imputed dataset.
7. `precision_variance`: Precision variances for each imputed dataset.
8. `accuracy_proportion`: Accuracy proportions for each imputed dataset.
9. `accuracy_variance`: Accuracy variances for each imputed dataset.
10. `AUROC_proportion`: AUROC proportions for each imputed dataset.
11. `AUROC_variance`: AUROC variances for each imputed dataset.

#### Example Data (First Few Rows)
Here’s an example of what the data might look like:

| dataset_name | sensitivity_proportion | sensitivity_variance | specificity_proportion | specificity_variance | precision_proportion | precision_variance | accuracy_proportion | accuracy_variance | AUROC_proportion | AUROC_variance |
|--------------|------------------------|----------------------|------------------------|----------------------|----------------------|--------------------|---------------------|-------------------|------------------|----------------|
| Dataset1     | 0.85                   | 0.002                | 0.90                   | 0.003                | 0.88                 | 0.002              | 0.87                | 0.003             | 0.89             | 0.002          |
| Dataset2     | 0.86                   | 0.0021               | 0.91                   | 0.0031               | 0.89                 | 0.0021             | 0.88                | 0.0031            | 0.90             | 0.0021         |
| Dataset3     | 0.84                   | 0.0019               | 0.89                   | 0.0029               | 0.87                 | 0.0019             | 0.86                | 0.0029            | 0.88             | 0.0019         |
| Dataset4     | 0.85                   | 0.0022               | 0.90                   | 0.0032               | 0.88                 | 0.0022             | 0.87                | 0.0032            | 0.89             | 0.0022         |
| Dataset5     | 0.86                   | 0.002                | 0.91                   | 0.003                | 0.89                 | 0.002              | 0.88                | 0.003             | 0.90             | 0.002          |

### Standard Error Computation

#### Rubin's Formula
To compute the Standard Error (SE) using Rubin's formula, the following steps are performed for each metric (sensitivity, specificity, precision, accuracy, and AUROC):

1. **Within-Imputation Variance (W)**:
   - Calculate the average of the variances within each imputed dataset.

2. **Between-Imputation Variance (B)**:
   - Calculate the variance of the proportions across the imputed datasets.

3. **Total Variance (T)**:
   - Combine W and B using the formula: \( T = W + \left(1 + \frac{1}{m}\right)B \)
   - Where \( m \) is the number of imputed datasets.

4. **Standard Error (SE)**:
   - Compute the standard error as the square root of the total variance: \( SE = \sqrt{T} \)

#### Example Computation
Assuming the above example data, the Python code provided will compute the standard error for each metric and output the results. Here’s how the output might look:

```plaintext
Standard Error for sensitivity: 0.045
Standard Error for specificity: 0.050
Standard Error for precision: 0.044
Standard Error for accuracy: 0.049
Standard Error for AUROC: 0.046
```

### Code Summary
The provided Python code performs the following tasks:

1. **Reads the CSV file**: The file is read using `pandas` to load the data into a DataFrame.
2. **Extracts relevant columns**: Proportions and variances for each metric are extracted into separate lists.
3. **Computes standard error**: For each metric, the within-imputation variance, between-imputation variance, total variance, and standard error are calculated using Rubin's formula.
4. **Prints results**: The standard error for each metric is printed.

This process ensures that the standard error for each metric is accurately computed, taking into account the variability both within and between the imputed datasets.
"""


import pandas as pd
import numpy as np

# Function to calculate within-imputation variance
def within_imputation_variance(variances):
    return np.mean(variances)

# Function to calculate between-imputation variance
def between_imputation_variance(proportions):
    means = np.array(proportions)
    mean_of_means = np.mean(means)
    between_var = np.var(means, ddof=1)  # Using ddof=1 to get sample variance
    return between_var

# Function to compute total variance using Rubin's formula
def rubins_total_variance(proportions, variances):
    m = len(proportions)
    W = within_imputation_variance(variances)
    B = between_imputation_variance(proportions)
    total_var = W + (1 + 1/m) * B
    return total_var

# Function to compute standard error using Rubin's formula
def standard_error(proportions, variances):
    total_var = rubins_total_variance(proportions, variances)
    return np.sqrt(total_var)

# Read the CSV file containing imputed datasets
csv_file_path = 'imputed_datasets.csv'
data = pd.read_csv(csv_file_path)

# Extract relevant columns for each metric
metrics = {
    'sensitivity': {
        'proportions': data['sensitivity_proportion'],
        'variances': data['sensitivity_variance']
    },
    'specificity': {
        'proportions': data['specificity_proportion'],
        'variances': data['specificity_variance']
    },
    'precision': {
        'proportions': data['precision_proportion'],
        'variances': data['precision_variance']
    },
    'accuracy': {
        'proportions': data['accuracy_proportion'],
        'variances': data['accuracy_variance']
    },
    'AUROC': {
        'proportions': data['AUROC_proportion'],
        'variances': data['AUROC_variance']
    }
}

# Compute and print standard error for each metric
for metric, values in metrics.items():
    se = standard_error(values['proportions'], values['variances'])
    print(f'Standard Error for {metric}: {se}')
