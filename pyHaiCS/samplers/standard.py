import jax
import jax.numpy as jnp
from tqdm import tqdm

def _single_chain_rejection_sampling(proposal, target, M, n_samples, key):
    """
    Single-chain rejection sampling.
    """
    sample_shape = proposal.sample(jax.random.PRNGKey(0)).shape
    samples = jnp.zeros((n_samples, *sample_shape))

    def cond_fn(state):
        i, _, _ = state
        return i < n_samples

    def body_fn(state):
        i, key, samples = state
        key, subkey1, subkey2 = jax.random.split(key, 3)
        x = proposal.sample(subkey1)
        u = jax.random.uniform(subkey2)
        p_accept = target(x) / (M * proposal.pdf(x))
        accept = u < p_accept

        # Only update samples if accepted
        samples = jax.lax.cond(
            accept,
            lambda s: s.at[i].set(x),
            lambda s: s,
            samples
        )

        i = jax.lax.cond(accept, lambda x: x + 1, lambda x: x, i)
        return i, key, samples

    init_state = (0, key, samples)
    _, _, final_samples = jax.lax.while_loop(cond_fn, body_fn, init_state)
    return final_samples

def rejection_sampling(proposal, target, M, n_samples, n_chains=4, RNG_key=42):
    """
    Multi-chain rejection sampling.
    """
    print("Running Rejection Sampler...")
    print("=" * 61)
    print(f"{'Num. Chains':^30}|{n_chains:^30}")
    print(f"{'Num. Samples':^30}|{n_samples:^30}")
    print(f"{'Proposal Bound M':^30}|{M:^30}")
    print("=" * 61)
    keys = jax.random.split(jax.random.PRNGKey(RNG_key), n_chains)
    vectorized_chain = jax.vmap(_single_chain_rejection_sampling, in_axes=(None, None, None, None, 0))
    samples = vectorized_chain(proposal, target, M, n_samples, keys)
    return samples

def _single_chain_importance_sampling(proposal, target, n_samples, key):
    """
    Single-chain importance sampling.
    """
    key, subkey = jax.random.split(key)
    x_samples = proposal.sample(subkey, shape=(n_samples,))
    weights = target(x_samples) / proposal.pdf(x_samples)
    weights = weights / jnp.sum(weights)  # Normalize weights
    return x_samples, weights

def importance_sampling(proposal, target, n_samples, n_chains=4, RNG_key=42):
    """
    Multi-chain importance sampling.
    """
    print("Running Importance Sampler...")
    print("=" * 61)
    print(f"{'Num. Chains':^30}|{n_chains:^30}")
    print(f"{'Num. Samples':^30}|{n_samples:^30}")
    print("=" * 61)
    keys = jax.random.split(jax.random.PRNGKey(RNG_key), n_chains)
    vectorized_chain = jax.vmap(_single_chain_importance_sampling, in_axes=(None, None, None, 0))
    samples, weights = vectorized_chain(proposal, target, n_samples, keys)
    return samples, weights