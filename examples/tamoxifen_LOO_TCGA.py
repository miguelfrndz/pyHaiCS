"""
Benchmarking Bayesian Logistic Regression on our Tamoxifen resistance data.
"""

import sys, os
sys.path.append('../')
import pyHaiCS as haics
import jax
jax.config.update("jax_enable_x64", True)
import numpy as np
import numpy.typing as npt
import pandas as pd
import jax.numpy as jnp
from sklearn.model_selection import StratifiedKFold, LeaveOneOut
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import make_scorer, precision_score, recall_score, matthews_corrcoef, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE

def load_data(dataset) -> tuple[npt.NDArray, npt.NDArray, npt.NDArray]:
    """
    Load the data from the specified dataset.
    """
    if dataset == 'TCGA':
        X = np.loadtxt('cancer_data/X_data_TCGA.txt', delimiter = ',')
        y = np.loadtxt('cancer_data/Y_data_TCGA.txt', delimiter = ',')
    elif dataset == 'METABRIC':
        data = pd.read_csv('cancer_data/patients_metabric.csv')
        X, y = data.iloc[:, 1:-1].values, data.iloc[:, -1].values
    else:
        raise ValueError('Invalid dataset. Choose between "TCGA" and "METABRIC"')
    priors = np.loadtxt('cancer_data/priors.txt', delimiter = ',')
    return X, y, priors

# Bayesian Logistic Regression model
@jax.jit
def model_fn(x, params):
    return jax.nn.sigmoid(jnp.matmul(x, params))

@jax.jit
def prior_fn(params):
    return jax.scipy.stats.norm.pdf(params)

@jax.jit
def log_prior_fn(params):
    return jnp.sum(jax.scipy.stats.norm.logpdf(params))

@jax.jit
def likelihood_fn(x, y, params):
    preds = model_fn(x, params)
    return jnp.prod(preds ** y * (1 - preds) ** (1 - y))

@jax.jit
def log_likelihood_fn(x, y, params):
    epsilon = 1e-7
    preds = model_fn(x, params)
    return jnp.sum(y * jnp.log(preds + epsilon) + (1 - y) * jnp.log(1 - preds + epsilon))

@jax.jit
def posterior_fn(x, y, params):
    return prior_fn(params) * likelihood_fn(x, y, params)

@jax.jit
def log_posterior_fn(x, y, params):
    return log_prior_fn(params) + log_likelihood_fn(x, y, params)

@jax.jit
def neg_posterior_fn(x, y, params):
    return -posterior_fn(x, y, params)

# Define a wrapper function to negate the log posterior
@jax.jit
def neg_log_posterior_fn(x, y, params):
    return -log_posterior_fn(x, y, params)

print(f"Running pyHaiCS v.{haics.__version__}")

# Load the data
X, y, priors = load_data(dataset = 'TCGA')

# Apply SMOTE to balance the classes
smote = SMOTE(sampling_strategy = 'minority', random_state = 42)
X, y = smote.fit_resample(X, y)

# Run cross-validation (either Stratified K-Fold or Leave-One-Out)
splitter = StratifiedKFold(n_splits = 5)
# splitter = LeaveOneOut()
n_splits = splitter.get_n_splits(y)

# Lists to store the metrics for each fold
precision_scores = []
recall_scores = []
accuracy_scores = []
sensitivity_scores = []
specificity_scores = []
f1_scores = []
mcc_scores = []

# Loop through each fold
for train_index, test_index in splitter.split(X, y):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    # Standardize the data
    scaler = StandardScaler()
    X_train = jnp.array(scaler.fit_transform(X_train))
    X_test = jnp.array(scaler.transform(X_test))

    # Add column of ones to the input data (for intercept terms)
    X_train = jnp.hstack([X_train, jnp.ones((X_train.shape[0], 1))])
    X_test = jnp.hstack([X_test, jnp.ones((X_test.shape[0], 1))])

    # Initialize the model parameters (includes intercept term)
    key = jax.random.PRNGKey(42)
    # Mean vector is the priors for the model parameters & 0 for the intercept term
    mean_vector = jnp.hstack([priors, 0])
    cov_mat = jnp.eye(X_train.shape[1]) * (2.5 ** 2)
    params = jax.random.multivariate_normal(key, mean_vector, cov_mat)

    # HMC for posterior sampling
    params_samples = haics.samplers.hamiltonian.HMC(params, 
                            potential_args = (X_train, y_train),                                           
                            n_samples = 5000, burn_in = 5000, 
                            step_size = 1e-3, n_steps = 100, 
                            potential = neg_log_posterior_fn,  
                            mass_matrix = jnp.eye(X_train.shape[1]), 
                            integrator = haics.integrators.VerletIntegrator(), 
                            RNG_key = 120)
    
    # HMC w/s-AIA adaptive scheme for posterior sampling
    # params_samples = haics.samplers.hamiltonian.sAIA(params,
    #                         potential_args = (X_train, y_train),
    #                         n_samples_tune = 1000, 
    #                         n_samples_check = 200,
    #                         n_samples_burn_in = 2000,
    #                         n_samples_prod = 5000,
    #                         potential = neg_log_posterior_fn,
    #                         mass_matrix = jnp.eye(X_train.shape[1]),
    #                         target_AR = 0.92, stage = 3, 
    #                         sensibility = 0.01, delta_step = 0.01, 
    #                         compute_freqs = True, sampler = "HMC", RNG_key = 42)
    
    ########################### GHMC ###########################
    # Momentum noise is randomly chosen between 0 and 1 (0 not included)
    # momentum_noise = jax.random.uniform(key, minval = 0.01, maxval = 1.0)

    # GHMC for posterior sampling
    # params_samples = haics.samplers.hamiltonian.GHMC(params, 
    #                             potential_args = (X_train, y_train),                                           
    #                             n_samples = 5000, burn_in = 5000, 
    #                             step_size = 1e-3, n_steps = 100, 
    #                             potential = neg_log_posterior_fn,  
    #                             mass_matrix = jnp.eye(X_train.shape[1]), 
    #                             momentum_noise = momentum_noise,
    #                             integrator = haics.integrators.VerletIntegrator(), 
    #                             RNG_key = 120)

    ############################################################
    # haics.utils.metrics.compute_metrics(params_samples, thres_estimator = 'var_trunc', normalize_ESS = True)

    # Average across chains (remove for s-AIA w/HMC)
    params_samples = jnp.mean(params_samples, axis = 0)

    # Make predictions using the samples
    preds = jax.vmap(lambda params: model_fn(X_test, params))(params_samples)
    y_test_pred = jnp.mean(preds, axis = 0)
    y_test_pred = (y_test_pred >= 0.5).astype("int")

    # Compute metrics
    precision_scores.append(precision_score(y_test, y_test_pred, zero_division = 1))
    recall_scores.append(recall_score(y_test, y_test_pred, zero_division = 1))
    sensitivity_scores.append(recall_score(y_test, y_test_pred, zero_division = 1))
    specificity_scores.append(recall_score(y_test, y_test_pred, pos_label = 0, zero_division = 1))
    f1_scores.append(f1_score(y_test, y_test_pred, zero_division = 1))
    mcc_scores.append(matthews_corrcoef(y_test, y_test_pred))
    accuracy_scores.append(accuracy_score(y_test, y_test_pred))

    # To avoid problems with compilation caches in memory, we perform some manual garbage collection :)
    jax.clear_caches()

# Calculate the global averages
average_precision = np.mean(precision_scores)
average_recall = np.mean(recall_scores)
average_sensitivity = np.mean(sensitivity_scores)
average_specificity = np.mean(specificity_scores)
average_f1 = np.mean(f1_scores)
average_mcc = np.mean(mcc_scores)
average_accuracy = np.mean(accuracy_scores)

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