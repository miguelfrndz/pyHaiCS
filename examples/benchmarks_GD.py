"""
Benchmarks for the Multivariate Gaussian Distribution model in HaiCS.
"""

import sys, os
sys.path.append('../')
import jax
import jax.numpy as jnp
import pandas as pd
import numpy as np
import pyHaiCS as haics
import matplotlib.pyplot as plt

GAUSSIAN_SIZE = 1000
    
# Multivariate Gaussian Distribution
@jax.jit
def potential_fn(inv_cov_matrix, params):
    return 1/2 * jnp.dot(jnp.dot(params.T, inv_cov_matrix), params)

print(f"Running pyHaiCS v.{haics.__version__}")

# Load the covariance matrix for the Gaussian distribution
filePath = os.path.join(os.path.dirname(__file__), f"../pyHaiCS/benchmarks/GD/D{GAUSSIAN_SIZE}_div_eig.txt")
eig_cov_matrix = np.loadtxt(filePath)
cov_matrix = jnp.diag(1/np.sqrt(eig_cov_matrix))
inv_cov_matrix = jnp.linalg.inv(cov_matrix)

# Initialize the model parameters
key = jax.random.PRNGKey(42)
key_HMC, key_GHMC = jax.random.split(key, 2)
mean_vector = jnp.zeros(GAUSSIAN_SIZE)
params = jax.random.multivariate_normal(key_HMC, mean_vector, cov_matrix)

params_samples_HMC = haics.samplers.hamiltonian.HMC(params, 
                            potential_args = (inv_cov_matrix, ),                                           
                            n_samples = 5000, burn_in = 5000, 
                            step_size = 1e-3, n_steps = 1000, 
                            potential = potential_fn,  
                            mass_matrix = jnp.eye(GAUSSIAN_SIZE), 
                            integrator = haics.integrators.VerletIntegrator(), 
                            RNG_key = 120, n_chains = 4)

params = jax.random.multivariate_normal(key_GHMC, mean_vector, cov_matrix)

params_samples_GHMC = haics.samplers.hamiltonian.GHMC(params, 
                            potential_args = (inv_cov_matrix, ),                                        
                            n_samples = 5000, burn_in = 5000, 
                            step_size = 1e-3, n_steps = 1000, 
                            potential = potential_fn,  
                            mass_matrix = jnp.eye(GAUSSIAN_SIZE), 
                            momentum_noise = 0.5,
                            integrator = haics.integrators.VerletIntegrator(), 
                            RNG_key = 120, n_chains = 4)

print("Results for HMC")
haics.utils.metrics.compute_metrics(params_samples_HMC, thres_estimator = 'var_trunc', normalize_ESS = True)
print("Results for GHMC")
haics.utils.metrics.compute_metrics(params_samples_GHMC, thres_estimator = 'var_trunc', normalize_ESS = True)

# Average across chains
params_samples_HMC = jnp.mean(params_samples_HMC, axis = 0)
params_samples_GHMC = jnp.mean(params_samples_GHMC, axis = 0)