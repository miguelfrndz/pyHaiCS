import jax
import jax.numpy as jnp
from tqdm import tqdm

@jax.jit
def Kinetic(p, mass_matrix):
    """
    Kinetic energy function.
    -------------------------
    Parameters:
        p (jax.Array): momentum
        mass_matrix (jax.Array): mass matrix
    -------------------------
    Returns:
        K (float): kinetic energy
    """
    K = 0.5 * jnp.dot(p, jnp.linalg.solve(mass_matrix, p))
    return K

def Hamiltonian(x, p, potential, mass_matrix):
    """
    Hamiltonian function.
    -------------------------
    Parameters:
        x (jax.Array): position
        p (jax.Array): momentum
        potential_grad (function): potential gradient
        mass_matrix (jax.Array): mass matrix
    -------------------------
    Returns:
        H (float): Hamiltonian
    """
    K = Kinetic(p, mass_matrix)
    U = potential(x)
    H = K + U
    return H

def _single_chain_HMC(x_init, n_samples, burn_in, step_size, n_steps, 
        potential, potential_grad, mass_matrix, integrator, key):
    """
    Single-Chain Hamiltonian Monte Carlo sampler.
    -------------------------
    Parameters:
        n_samples (int): number of samples
        burn_in (int): burn-in samples
        step_size (float): step size
        n_steps (int): number of integration steps
        potential (function): Hamiltonian potential
        potential_grad (function): Hamiltonian potential gradient
        mass_matrix (jax.Array): mass matrix
        integrator (object): integrator object
    -------------------------
    Returns:
        samples (jax.Array): samples
    """
    samples = []
    x = x_init
    for n in tqdm(range(n_samples + burn_in)):
        key, subkey = jax.random.split(key)
        # Initial momentum (gaussian), shape given by mass matrix
        p = jax.random.multivariate_normal(subkey, jnp.zeros(x.shape[0]), mass_matrix)
        # Integrate Hamiltonian dynamics
        x_prop, p_prop = integrator.integrate(x, p, potential_grad, n_steps, mass_matrix, step_size)
        # Computer enery error
        delta_H = Hamiltonian(x_prop, p_prop, potential, mass_matrix) - Hamiltonian(x, p, potential, mass_matrix)
        # Metropolis-Hastings acceptance
        accept = jax.random.uniform(subkey) < jnp.exp(-delta_H)
        x = jax.lax.cond(accept, lambda _: x_prop, lambda _: x, operand=None)
        if n >= burn_in:
            samples.append(x)
    samples = jnp.stack(samples, axis=0)
    return samples

def HMC(x_init, potential_args, n_samples, burn_in, step_size, n_steps, 
        potential, mass_matrix, integrator, n_chains=4, RNG_key=42):
    """
    Multi-Chain Hamiltonian Monte Carlo sampler.
    -------------------------
    Parameters:
        n_samples (int): number of samples
        burn_in (int): burn-in samples
        step_size (float): step size
        n_steps (int): number of integration steps
        potential (function): Hamiltonian potential
        mass_matrix (jax.Array): mass matrix
        integrator (object): integrator object
        n_chains (int): number of chains
        RNG_key (int): random number generator key
    -------------------------
    Returns:
        samples (jax.Array): samples
    """
    print("Running HMC sampler...")
    print("="*61)
    print(f"{'Num. Chains':^30}|{n_chains:^30}")
    print(f"{'Num. Samples':^30}|{n_samples:^30}")
    print(f"{'Num. Burn-In Iterations':^30}|{burn_in:^30}")
    print(f"{'Step-Size':^30}|{step_size:^30}")
    print(f"{'Num. Integration Steps':^30}|{n_steps:^30}")
    print("="*61)
    keys = jax.random.split(jax.random.PRNGKey(RNG_key), n_chains)
    x_init_repeated = jnp.repeat(x_init[None, :], n_chains, axis = 0)
    potential = jax.tree_util.Partial(potential, *potential_args)
    potential_grad = jax.grad(potential)
    vectorized_chain = jax.vmap(_single_chain_HMC, in_axes=(0, None, None, None, None, None, None, None, None, 0))
    samples = vectorized_chain(x_init_repeated, n_samples, burn_in, step_size, n_steps, potential, potential_grad, mass_matrix, integrator, keys)
    return samples

def GHMC():
    pass

def MMHMC():
    pass