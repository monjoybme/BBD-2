import os
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, roc_auc_score, precision_score, recall_score, accuracy_score
from scipy.stats import norm
from sklearn.utils import resample

# Function to calculate confidence interval for a proportion
def calculate_confidence_interval(proportion, n, confidence=0.95):
    """Returns the confidence interval for a proportion using the normal approximation."""
    z = norm.ppf(1 - (1 - confidence) / 2)
    se = np.sqrt(proportion * (1 - proportion) / n)
    lower_bound = proportion - z * se
    upper_bound = proportion + z * se
    return lower_bound, upper_bound

# Function to calculate variance of a proportion
def calculate_variance(proportion, n):
    """Returns the variance of a proportion."""
    return proportion * (1 - proportion) / n

# Function to calculate the metrics and their statistics
def calculate_metrics(y_true, y_pred_proba, threshold=0.5, n_bootstrap=1000):
    # Get predictions
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

    # Calculate sensitivity, specificity, precision, accuracy
    sensitivity = tp / (tp + fn)  # True Positive Rate
    specificity = tn / (tn + fp)  # True Negative Rate
    precision = tp / (tp + fp)  # Positive Predictive Value
    accuracy = (tp + tn) / (tp + tn + fp + fn)  # Accuracy

    # Calculate AUROC (Area Under Receiver Operating Characteristic Curve)
    auroc = roc_auc_score(y_true, y_pred_proba)

    # Bootstrap sampling for AUROC confidence interval and variance
    auroc_bootstrap = []
    for _ in range(n_bootstrap):
        # Resample with replacement
        X_resampled, y_resampled = resample(y_pred_proba, y_true, random_state=None)
        auroc_resampled = roc_auc_score(y_resampled, X_resampled)
        auroc_bootstrap.append(auroc_resampled)
    
    auroc_var = np.var(auroc_bootstrap)  # Variance of AUROC from bootstrap samples
    auroc_ci_lower, auroc_ci_upper = np.percentile(auroc_bootstrap, [2.5, 97.5])  # 95% CI

    # Sample size
    n = len(y_true)

    # Calculate confidence intervals and variances for other metrics
    sensitivity_ci = calculate_confidence_interval(sensitivity, n)
    specificity_ci = calculate_confidence_interval(specificity, n)
    precision_ci = calculate_confidence_interval(precision, n)
    accuracy_ci = calculate_confidence_interval(accuracy, n)

    sensitivity_var = calculate_variance(sensitivity, n)
    specificity_var = calculate_variance(specificity, n)
    precision_var = calculate_variance(precision, n)
    accuracy_var = calculate_variance(accuracy, n)

    return {
        "sensitivity": (sensitivity, sensitivity_var, sensitivity_ci),
        "specificity": (specificity, specificity_var, specificity_ci),
        "precision": (precision, precision_var, precision_ci),
        "accuracy": (accuracy, accuracy_var, accuracy_ci),
        "auroc": (auroc, auroc_var, (auroc_ci_lower, auroc_ci_upper))
    }

# Function to process all CSV files in the folder
def process_folder(input_folder, output_file):
    # Prepare results
    results = []

    # Loop through all files in the folder
    for filename in os.listdir(input_folder):
        if filename.endswith(".csv"):
            file_path = os.path.join(input_folder, filename)
            # Load the CSV file
            df = pd.read_csv(file_path)
            
            # Assuming the columns are named 'y_test' and 'y_pred_proba'
            y_test = df['y_test'].values
            y_pred_proba = df['y_pred_proba'].values

            # Calculate the metrics
            metrics = calculate_metrics(y_test, y_pred_proba)

            # Store the results
            result_row = {
                "filename": filename,
                "sensitivity_proportion": metrics["sensitivity"][0],
                "sensitivity_variance": metrics["sensitivity"][1],
                "sensitivity_ci_lower": metrics["sensitivity"][2][0],
                "sensitivity_ci_upper": metrics["sensitivity"][2][1],
                "specificity_proportion": metrics["specificity"][0],
                "specificity_variance": metrics["specificity"][1],
                "specificity_ci_lower": metrics["specificity"][2][0],
                "specificity_ci_upper": metrics["specificity"][2][1],
                "precision_proportion": metrics["precision"][0],
                "precision_variance": metrics["precision"][1],
                "precision_ci_lower": metrics["precision"][2][0],
                "precision_ci_upper": metrics["precision"][2][1],
                "accuracy_proportion": metrics["accuracy"][0],
                "accuracy_variance": metrics["accuracy"][1],
                "accuracy_ci_lower": metrics["accuracy"][2][0],
                "accuracy_ci_upper": metrics["accuracy"][2][1],
                "auroc_proportion": metrics["auroc"][0],
                "auroc_variance": metrics["auroc"][1],
                "auroc_ci_lower": metrics["auroc"][2][0],
                "auroc_ci_upper": metrics["auroc"][2][1]
            }
            results.append(result_row)

    # Convert results to DataFrame and save as Excel
    results_df = pd.DataFrame(results)
    results_df.to_excel(output_file, index=False)

# Example usage
input_folder = "/Users/saham2/Downloads/BBD_revision_08January2025/AnalysisUsing_esr_contrlq_8Jan2025/RevisonStudyUsing_esr_contrlq_8jan2025/All_predictionFiles8Jan2025"  # Replace with your folder path
output_file = "metrics_results.xlsx"  # Desired output file name
process_folder(input_folder, output_file)

print(f"Metrics saved to {output_file}")
