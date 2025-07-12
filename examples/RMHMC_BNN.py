"""
This script demonstrates the use of Riemannian Manifold Hamiltonian Monte Carlo (RMHMC) to sample from a simple banana-shaped posterior distribution (https://arxiv.org/pdf/2111.09995).
"""
import jax
import jax.numpy as jnp
import pandas as pd
import matplotlib.pyplot as plt

import sys, os
sys.path.append('../')

import pyHaiCS as haics
from pyHaiCS.utils.hamiltonian import fisher_metric, generalized_fisher_metric
from pyHaiCS.integrators.integrators import RMHMCVerletIntegrator

# 1. Potential function (negative log-posterior) for the banana-shaped distribution
@jax.jit
def potential_fn(y, sigma_y, sigma_params, params):
    log_likelihood_term = 1/(2 * sigma_y ** 2) * jnp.sum((y - params[0] - params[1]**2) ** 2)
    log_prior_term = 1/(2 * sigma_params ** 2) * (params[0] ** 2 + params[1] ** 2)
    return log_likelihood_term + log_prior_term

# 2. Log-likelihood function (for the Fisher metric)
@jax.jit
def log_likelihood_fn(y, sigma_y, params):
    return -0.5 / sigma_y**2 * jnp.sum((y - params[0] - params[1]**2) ** 2)

# 3. Negative log-prior function (for the prior curvature)
@jax.jit
def neg_log_prior_fn(sigma_params, params):
    return 1/(2 * sigma_params ** 2) * jnp.sum(params**2)

print(f"Running pyHaiCS v.{haics.__version__}")

# Load the values y for the banana-shaped distribution
filePath = os.path.join(os.path.dirname(__file__), f"../pyHaiCS/benchmarks/BNN/Banana_100.txt")
y = pd.read_table(filePath, header = None, sep = '\\s+').values.reshape(-1)
y = jnp.array(y)

# Initialize the model parameters
key = jax.random.PRNGKey(42)
key_HMC, key_GHMC = jax.random.split(key, 2)
mean_vector = jnp.zeros(2)
cov_mat = jnp.eye(2)
params = jax.random.multivariate_normal(key_HMC, mean_vector, cov_mat)
sigma_y, sigma_params = 2, 1

# === Run RMHMC ===
dim = 2
key = jax.random.PRNGKey(0)

samples = haics.samplers.hamiltonian.RMHMC(
        x_init = params,
        potential_args = (y, sigma_y, sigma_params),
        n_samples = 5000,
        burn_in = 2000,
        step_size = 2.9e-3,
        n_steps = 500,
        potential = potential_fn,
        metric = generalized_fisher_metric(log_likelihood_fn,
                                         neg_log_prior_fn, 
                                         (y, sigma_y), 
                                         (sigma_params,)),
        integrator = RMHMCVerletIntegrator(),
        n_chains = 1
)

# === Visualize Sampling ===

samples = samples.squeeze()
mean_est = jnp.mean(samples, axis=0)
cov_est = jnp.cov(samples.T)

print("Posterior Mean Estimate:", mean_est)
print("Posterior Covariance Estimate:\n", cov_est)

# === Plot true density contours ===
# Create a meshgrid over parameter space
theta1_vals = jnp.linspace(-2, 2, 2000)
theta2_vals = jnp.linspace(-2, 2, 2000)
Theta1, Theta2 = jnp.meshgrid(theta1_vals, theta2_vals)

# Compute unnormalized log posterior
def unnormalized_posterior(theta1, theta2):
    params = jnp.array([theta1, theta2])
    return -potential_fn(y, sigma_y, sigma_params, params)

log_density = jnp.vectorize(unnormalized_posterior)(Theta1, Theta2)
log_density = log_density - jnp.max(log_density)  # for numerical stability
density = jnp.exp(log_density)

plt.figure(figsize=(6, 6))
# Plot filled contours for density above a threshold (e.g., 0.1 of peak)
plt.contourf(Theta1, Theta2, density, levels=20, alpha=0.3)
# Overlay RMHMC samples
plt.plot(samples[:, 0], samples[:, 1], '.', alpha=0.2, label="RMHMC Samples", markersize=2)
plt.xlabel(r"$\theta_1$")
plt.ylabel(r"$\theta_2$")
plt.title("Banana-Shaped Density (Shaded) with RMHMC Samples")
plt.grid(True)
plt.axis('equal')
plt.xlim(-1.75, 1.75)
plt.ylim(-1.75, 1.75)
plt.tight_layout()
plt.savefig("banana_rmhmc_samples.png", dpi=200)
plt.show()