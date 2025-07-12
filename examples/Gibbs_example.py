import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.append('../')
from pyHaiCS.samplers.basic_mcmc import gibbs_sampling
from scipy.stats import multivariate_normal

def conditional_sampler(state, dim_index, key):
    """
    Conditional sampler for a 2D Gaussian distribution.
    """
    rho = 0.8
    sigma_sq = 1.0
    mu = 0.0
    other_dim = 1 - dim_index
    other_value = state[other_dim]

    conditional_mu = mu + rho * (other_value - mu)
    conditional_sigma = jnp.sqrt(sigma_sq * (1 - rho**2))

    return jax.random.normal(key) * conditional_sigma + conditional_mu

dim = 2
n_samples = 1000
n_chains = 4
initial_states = jnp.zeros((n_chains, dim))

samples = gibbs_sampling(
    conditional_sampler=conditional_sampler,
    initial_states=initial_states,
    n_samples=n_samples,
    n_chains=n_chains,
    RNG_key=0
)
samples = np.array(samples.reshape(-1, 2))

# Target distribution parameters
mean = np.array([0, 0])
cov = np.array([[1, 0.8], [0.8, 1]])

# Create grid for contour plot
x, y = np.mgrid[-3:3:.01, -3:3:.01]
pos = np.dstack((x, y))
rv = multivariate_normal(mean, cov)

pdf = rv.pdf(pos)
plt.contour(x, y, pdf, levels=10, linewidths=1.5, linestyles='dashed', cmap='summer')
plt.scatter(samples[:, 0], samples[:, 1], alpha=0.3, s=10, label="Gibbs Samples", color='black')
plt.title("Samples from 2D Gaussian via Gibbs Sampling")
plt.xlabel("X0")
plt.ylabel("X1")
plt.axis("equal")
plt.grid(True)
plt.show()