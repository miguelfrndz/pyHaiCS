"""
Benchmarking Shallow ML models on our Tamoxifen resistance data.
"""

import sys, os
sys.path.append('../')
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, precision_score, recall_score, matthews_corrcoef, accuracy_score, f1_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

def load_data() -> tuple:
    """
    Load the data.
    """
    X = np.loadtxt('cancer_data/X_data.txt', delimiter = ',')
    y = np.loadtxt('cancer_data/Y_data.txt', delimiter = ',')
    priors = np.loadtxt('cancer_data/priors.txt', delimiter = ',')
    return X, y, priors

# Load the data
X, y, priors = load_data()

# Run stratified cross validation
# splitter = StratifiedKFold(n_splits = 5)
splitter = LeaveOneOut()

# Lists to store the metrics for each fold
precision_scores = []
recall_scores = []
accuracy_scores = []
sensitivity_scores = []
specificity_scores = []
f1_scores = []
mcc_scores = []

for train_index, test_index in splitter.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Standardize the data
    scaler = StandardScaler()
    X_train = np.array(scaler.fit_transform(X_train))
    X_test = np.array(scaler.transform(X_test))

    # Logistic Regression using Scikit-Learn (for comparison)
    # clf = LogisticRegression(random_state = 42)
    clf = RandomForestClassifier(random_state = 42)
    clf.fit(X_train, y_train)
    y_test_pred = clf.predict(X_test)

    # Compute metrics
    precision_scores.append(precision_score(y_test, y_test_pred, zero_division = 1))
    recall_scores.append(recall_score(y_test, y_test_pred, zero_division = 1))
    sensitivity_scores.append(recall_score(y_test, y_test_pred, zero_division = 1))
    specificity_scores.append(recall_score(y_test, y_test_pred, pos_label = 0, zero_division = 1))
    f1_scores.append(f1_score(y_test, y_test_pred, zero_division = 1))
    mcc_scores.append(matthews_corrcoef(y_test, y_test_pred))
    accuracy_scores.append(accuracy_score(y_test, y_test_pred))

# Calculate the global averages
average_precision = np.mean(precision_scores)
average_recall = np.mean(recall_scores)
average_sensitivity = np.mean(sensitivity_scores)
average_specificity = np.mean(specificity_scores)
average_f1 = np.mean(f1_scores)
average_mcc = np.mean(mcc_scores)
average_accuracy = np.mean(accuracy_scores)

n_splits = splitter.get_n_splits(y)

# Create a DataFrame to display the results
metrics_df = pd.DataFrame({
    'Fold': range(1, n_splits + 1),
    'Precision': precision_scores,
    'Recall': recall_scores,
    'Sensitivity': sensitivity_scores,
    'Specificity': specificity_scores,
    'F1': f1_scores,
    'MCC': mcc_scores,
    'Accuracy': accuracy_scores
})

# Adding global averages
metrics_df.loc['Average'] = ['-', average_precision, average_recall, average_sensitivity, average_specificity, average_f1, average_mcc, average_accuracy]
print(metrics_df)