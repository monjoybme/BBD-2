"""
Python code to compute the confidence interval for the Area Under the Curve (AUC) of a Receiver Operating Characteristic (ROC) curve using bootstrapping
"""

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.utils import resample

def bootstrap_auc(y_true, y_score, n_bootstrap=1000, confidence_level=0.95):
    aucs = []
    n_samples = len(y_true)
    for _ in range(n_bootstrap):
        indices = np.random.choice(n_samples, n_samples, replace=True)
        bootstrapped_y_true = y_true[indices]
        bootstrapped_y_score = y_score[indices]
        auc = roc_auc_score(bootstrapped_y_true, bootstrapped_y_score)
        aucs.append(auc)
    
    # Calculate confidence interval
    alpha = (1 - confidence_level) / 2
    lower_percentile = 100 * alpha
    upper_percentile = 100 * (1 - alpha)
    lower_bound = np.percentile(aucs, lower_percentile)
    upper_bound = np.percentile(aucs, upper_percentile)
    
    return lower_bound, upper_bound

# Example usage:
# Assuming y_true and y_score are your true labels and predicted scores
# y_true = ...
# y_score = ...
lower_bound, upper_bound = bootstrap_auc(y_true, y_score)
print("95% Confidence Interval for AUC:", (lower_bound, upper_bound))
