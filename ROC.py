"""
Python code to compute ROC and CI
"""

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.utils import resample
import matplotlib.pyplot as plt

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
# Read data from CSV
data = pd.read_csv('path_to_your_file.csv')

# Assuming the CSV has columns 'y_true' and 'y_score'
y_true = data['y_true'].values
y_score = data['y_score'].values

# Calculate AUC confidence intervals
lower_bound, upper_bound = bootstrap_auc(y_true, y_score)
print("95% Confidence Interval for AUC:", (lower_bound, upper_bound))

# Plot ROC curve
fpr, tpr, _ = roc_curve(y_true, y_score)
roc_auc = roc_auc_score(y_true, y_score)

plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

