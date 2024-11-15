"""
Benchmarks for the BNN model in HaiCS.
"""

import sys, os
sys.path.append('../')
import jax
import jax.numpy as jnp
import pandas as pd
import pyHaiCS as haics
import matplotlib.pyplot as plt
    
# Banana-shaped Distribution
@jax.jit
def potential_fn(y, sigma_y, sigma_params, params):
    return 1/(2 * sigma_y ** 2) * jnp.sum((y - params[0] - params[1]**2) ** 2) + 1/(2 * sigma_params ** 2) * (params[0] ** 2 + params[1] ** 2)

print(f"Running pyHaiCS v.{haics.__version__}")

# Load the values y for the banana-shaped distribution
filePath = os.path.join(os.path.dirname(__file__), f"../pyHaiCS/benchmarks/BNN/Banana_100.txt")
y = pd.read_table(filePath, header = None, sep = '\\s+').values.reshape(-1)

# Initialize the model parameters
key = jax.random.PRNGKey(42)
key_HMC, key_GHMC = jax.random.split(key, 2)
# y = jax.random.normal(key, shape = (100,))
mean_vector = jnp.zeros(2)
cov_mat = jnp.eye(2)
params = jax.random.multivariate_normal(key_HMC, mean_vector, cov_mat)
sigma_y, sigma_params = 2, 1

params_samples_HMC = haics.samplers.hamiltonian.HMC(params, 
                            potential_args = (y, sigma_y, sigma_params),                                           
                            n_samples = 5000, burn_in = 5000, 
                            step_size = 1/9, n_steps = 14, 
                            potential = potential_fn,  
                            mass_matrix = jnp.eye(2), 
                            integrator = haics.integrators.VerletIntegrator(), 
                            RNG_key = 120, n_chains = 3)

params = jax.random.multivariate_normal(key_GHMC, mean_vector, cov_mat)

params_samples_GHMC = haics.samplers.hamiltonian.GHMC(params, 
                            potential_args = (y, sigma_y, sigma_params),                                        
                            n_samples = 5000, burn_in = 5000, 
                            step_size = 1/9, n_steps = 14, 
                            potential = potential_fn,  
                            mass_matrix = jnp.eye(2), 
                            momentum_noise = 0.5,
                            integrator = haics.integrators.VerletIntegrator(), 
                            RNG_key = 120, n_chains = 3)

print("Results for HMC")
haics.utils.metrics.compute_metrics(params_samples_HMC, thres_estimator = 'var_trunc', normalize_ESS = True)
print("Results for GHMC")
haics.utils.metrics.compute_metrics(params_samples_GHMC, thres_estimator = 'var_trunc', normalize_ESS = True)

# Average across chains
# params_samples_HMC = jnp.mean(params_samples_HMC, axis = 0)
# params_samples_GHMC = jnp.mean(params_samples_GHMC, axis = 0)

# Plot theta1 vs theta2 to show the banana distribution
fig, axs = plt.subplots(1, 2, figsize=(20, 6))

# Plot theta1 vs theta2 to show the banana distribution for HMC
for i in range(params_samples_HMC.shape[0]):
    axs[0].scatter(params_samples_HMC[i, :, 0], params_samples_HMC[i, :, 1], alpha=0.5, edgecolor='k', s=50, label=f'Chain {i+1}')
axs[0].set_title('Banana Distribution: Theta1 vs Theta2 (HMC)', fontsize=16)
axs[0].set_xlabel('Theta1', fontsize=14)
axs[0].set_ylabel('Theta2', fontsize=14)
axs[0].grid(True)
axs[0].legend()

# Plot theta1 vs theta2 to show the banana distribution for GHMC
for i in range(params_samples_GHMC.shape[0]):
    axs[1].scatter(params_samples_GHMC[i, :, 0], params_samples_GHMC[i, :, 1], alpha=0.5, edgecolor='k', s=50, label=f'Chain {i+1}')
axs[1].set_title('Banana Distribution: Theta1 vs Theta2 (GHMC)', fontsize=16)
axs[1].set_xlabel('Theta1', fontsize=14)
axs[1].set_ylabel('Theta2', fontsize=14)
axs[1].grid(True)
axs[1].legend()

plt.tight_layout()
plt.show()