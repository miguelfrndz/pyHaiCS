"""
This script demonstrates how to use the pyHaiCS library to compute effective sample sizes (ESS) using Geyer's method, multiESS, and codaESS on samples generated from a simple rejection sampling algorithm.
"""

import jax
import jax.numpy as jnp

import sys
sys.path.append('../')

import pyHaiCS as haics
from pyHaiCS.utils.metrics import geyerESS, multiESS, codaESS

# Define the target and proposal PDFs using JAX
def target_pdf(x):
    return jnp.sin(2 * jnp.pi * x) + 1

def proposal_pdf(x):
    return jnp.where((x >= 0) & (x <= 1), 1.0, 0.0)

# Parameters
N_SAMPLES = 10_000
N_CHAINS = 4
k = 2.0

def rejection_sample(key):
    key1, key2 = jax.random.split(key)
    u = jax.random.uniform(key1, shape=(N_SAMPLES,))
    y = jax.random.uniform(key2, shape=(N_SAMPLES,))
    accept_prob = target_pdf(y) / (k * proposal_pdf(y))
    accepted = u < accept_prob
    samples = y[accepted]
    return samples

# Generate PRNG keys for all chains
main_key = jax.random.PRNGKey(0)
keys = jax.random.split(main_key, N_CHAINS)

# Vectorize rejection sampling across chains
all_samples = list(map(rejection_sample, keys))

# Pad chains to the shortest one to align shapes
min_len = min(s.shape[0] for s in all_samples)
aligned_samples = jnp.stack([s[:min_len] for s in all_samples])

# Reshape for ESS: (num_chains, num_samples, dim)
aligned_samples = aligned_samples.reshape(N_CHAINS, min_len, 1)

ess_values = geyerESS(aligned_samples, normalize = False) / N_SAMPLES
multiESS_values = multiESS(aligned_samples, normalize = False) / N_SAMPLES
codaESS_values = codaESS(aligned_samples, normalize = False, method = 'monotone-sequence') / N_SAMPLES

print("Effective Sample Sizes:")
print("\t - Geyer's ESS:", ess_values.flatten())
print("\t - Multi ESS:", multiESS_values.flatten())
print("\t - CODA ESS:", codaESS_values.flatten())