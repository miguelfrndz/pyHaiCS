"""
Benchmarks for the BLR model in HaiCS with different binary classification datasets.
"""

import sys, os
sys.path.append('../')
import jax
import argparse
import jax.numpy as jnp
import pandas as pd
import pyHaiCS as haics
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

def load_dataset(dataset : str) -> tuple:
    """
    Load the specified dataset.
    """
    valid_datasets = ["german", "musk", "secom", "sonar"]
    if dataset not in valid_datasets:
        raise ValueError(f"Invalid dataset name. Please choose one of: {valid_datasets}")
    filePath = os.path.join(os.path.dirname(__file__), f"../pyHaiCS/benchmarks/BLR/{dataset}.txt")
    data = pd.read_table(filePath, header = None, sep = '\\s+').values
    X, y = data[:, :-1], data[:, -1]
    return X, y

def baseline_classifier(X_train, y_train, X_test, y_test):
    clf = LogisticRegression(random_state = 42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy (w/ Scikit-Learn): {accuracy}\n")
    
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

# Load the specified dataset
parser = argparse.ArgumentParser(description="Run BLR benchmarks with different datasets.")
parser.add_argument('dataset', type=str, help="Name of the dataset to use.")
args = parser.parse_args()

# Load the specified dataset
X, y = load_dataset(args.dataset)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42, stratify = y)

# Standardize the data & convert to jax arrays
scaler = StandardScaler()
X_train = jnp.array(scaler.fit_transform(X_train))
X_test = jnp.array(scaler.transform(X_test))

# Add column of ones to the input data (for intercept terms)
X_train = jnp.hstack([X_train, jnp.ones((X_train.shape[0], 1))])
X_test = jnp.hstack([X_test, jnp.ones((X_test.shape[0], 1))])

# Initialize the model parameters (includes intercept term)
key = jax.random.PRNGKey(42)
mean_vector = jnp.zeros(X_train.shape[1])
cov_mat = jnp.eye(X_train.shape[1])
params = jax.random.multivariate_normal(key, mean_vector, cov_mat)

# HMC for posterior sampling
params_samples = haics.samplers.hamiltonian.HMC(params, 
                            potential_args = (X_train, y_train),                                           
                            n_samples = 1000, burn_in = 200, 
                            step_size = 1e-3, n_steps = 100, 
                            potential = neg_log_posterior_fn,  
                            mass_matrix = jnp.eye(X_train.shape[1]), 
                            integrator = haics.integrators.VerletIntegrator(), 
                            RNG_key = 120)

haics.utils.metrics.compute_metrics(params_samples, thres_estimator = 'var_trunc', normalize_ESS = True)

# Average across chains
params_samples = jnp.mean(params_samples, axis = 0)

# Make predictions using the samples
preds = jax.vmap(lambda params: model_fn(X_test, params))(params_samples)
mean_preds = jnp.mean(preds, axis = 0)
mean_preds = mean_preds > 0.5

# Evaluate the model
accuracy = jnp.mean(mean_preds == y_test)
print(f"Accuracy (w/ HMC Sampling): {accuracy}\n")

# Compare with baseline classifier (scikit-learn)
baseline_classifier(X_train, y_train, X_test, y_test)