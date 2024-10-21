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

def Hamiltonian(x, p, potential, potential_args, mass_matrix):
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
    U = potential(x, *potential_args)
    H = K + U
    return H

def HMC(x_init, potential_args, n_samples, burn_in, step_size, n_steps, potential, mass_matrix, integrator, n_chains = 4, RNG_key = 42):
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
    -------------------------
    Returns:
        samples (jax.Array): samples
    """
    print("Running HMC sampler...")
    key = jax.random.PRNGKey(RNG_key)
    potential = jax.jit(potential)
    potential_grad = jax.grad(potential)
    samples = []
    x = x_init
    for n in tqdm(range(n_samples + burn_in)):
        # Initial momentum (gaussian), shape given by mass matrix
        p = jax.random.multivariate_normal(key, jnp.zeros(x.shape[0]), mass_matrix)
        # Integrate Hamiltonian dynamics
        x_prop, p_prop = integrator.integrate(x, p, potential_grad, potential_args, n_steps, mass_matrix, step_size)
        # Computer enery error
        delta_H = Hamiltonian(x_prop, p_prop, potential, potential_args, mass_matrix) - Hamiltonian(x, p, potential, potential_args, mass_matrix)
        # Metropolis-Hastings acceptance
        if jax.random.uniform(jax.random.PRNGKey(0)) < jnp.exp(-delta_H):
            x = x_prop
        if n >= burn_in:
            samples.append(x)
    samples = jnp.stack(samples, axis=0)
    return samples

def GHMC():
    pass

def MMHMC():
    pass