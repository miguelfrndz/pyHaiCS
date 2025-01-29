"""
Implementations of Basic Markov Chain Monte Carlo Methods
"""

import jax
import jax.numpy as jnp
from tqdm import tqdm

def _single_chain_RWMH(x_init, n_samples, burn_in, step_size, potential, key):
    """
    Single-Chain Random-Walk Metropolis-Hastings (RWMH) sampler.
    -------------------------
    Parameters:
        x_init (jax.Array): initial position
        n_samples (int): number of samples
        burn_in (int): burn-in samples
        step_size (float): step size
        potential (function): Hamiltonian potential
    -------------------------
    Returns:
        samples (jax.Array): samples
    """
    x = x_init
    samples = jnp.zeros((n_samples, *x.shape))
    for n in tqdm(range(n_samples + burn_in)):
        key, subkey_proposal, subkey_acceptance = jax.random.split(key, 3)
        # Propose new position
        x_prop = x + step_size * jax.random.normal(subkey_proposal, shape=x.shape)
        # Compute energy difference
        delta_U = potential(x_prop) - potential(x)
        # Metropolis-Hastings acceptance
        accept = jax.random.uniform(subkey_acceptance) < jnp.exp(-delta_U)
        x = jax.lax.cond(accept, lambda _: x_prop, lambda _: x, operand=None)
        if n >= burn_in:
            samples = samples.at[n - burn_in].set(x)
    return samples

def RWMH(x_init, potential_args, n_samples, burn_in, step_size, potential, n_chains=4, RNG_key=42):
    """
    Multi-Chain Random-Walk Metropolis-Hastings (RWMH) sampler.
    -------------------------
    Parameters:
        x_init (jax.Array): initial position
        potential_args (tuple): arguments for Hamiltonian potential
        n_samples (int): number of samples
        burn_in (int): burn-in samples
        step_size (float): step-size
        potential (function): Hamiltonian potential
        n_chains (int): number of chains
        RNG_key (int): random number generator key
    -------------------------
    Returns:
        samples (jax.Array): samples
    """
    print("Running RWMH sampler...")
    print("="*61)
    print(f"{'Num. Chains':^30}|{n_chains:^30}")
    print(f"{'Num. Samples':^30}|{n_samples:^30}")
    print(f"{'Num. Burn-In Iterations':^30}|{burn_in:^30}")
    print(f"{'Step-Size':^30}|{step_size:^30}")
    print("="*61)
    keys = jax.random.split(jax.random.PRNGKey(RNG_key), n_chains)
    x_init_repeated = jnp.repeat(x_init[None, :], n_chains, axis=0)
    potential = jax.jit(jax.tree_util.Partial(potential, *potential_args))
    vectorized_chain = jax.vmap(_single_chain_RWMH, in_axes=(0, None, None, None, None, 0))
    samples = vectorized_chain(x_init_repeated, n_samples, burn_in, step_size, potential, keys)
    return samples