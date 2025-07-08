import jax
import jax.numpy as jnp
from tqdm import tqdm
from functools import partial
from ..integrators.integrators import VerletIntegrator, VV_2, ME_2, VV_3, ME_3, MSSI_2, MSSI_3, Integrator
from ..utils.metrics import acceptance_rate
from ..utils.hamiltonian import Hamiltonian

def _single_chain_HMC(x_init, n_samples, burn_in, step_size, n_steps, 
        potential, potential_grad, mass_matrix, integrator, key):
    """
    Single-Chain Hamiltonian Monte-Carlo (HMC) sampler.
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
        potential, mass_matrix, integrator = VerletIntegrator(), n_chains = 4, RNG_key = 42):
    """
    Multi-Chain Hamiltonian Monte-Carlo (HMC) sampler.
    -------------------------
    Parameters:
        x_init (jax.Array): initial position
        potential_args (tuple): arguments for Hamiltonian potential
        n_samples (int): number of samples
        burn_in (int): burn-in samples
        step_size (float): step-size
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
    potential = jax.jit(jax.tree_util.Partial(potential, *potential_args))
    potential_grad = jax.jit(jax.grad(potential))
    vectorized_chain = jax.vmap(_single_chain_HMC, in_axes=(0, None, None, None, None, None, None, None, None, 0))
    samples = vectorized_chain(x_init_repeated, n_samples, burn_in, step_size, n_steps, potential, potential_grad, mass_matrix, integrator, keys)
    return samples

def _single_chain_GHMC(x_init, n_samples, burn_in, step_size, n_steps, 
        potential, potential_grad, mass_matrix, momentum_noise, integrator, 
        key, metropolize):
    """
    Single-Chain Generalized Hamiltonian Monte-Carlo (GHMC) sampler.
    -------------------------
    Parameters:
        n_samples (int): number of samples
        burn_in (int): burn-in samples
        step_size (float): step size
        n_steps (int): number of integration steps
        potential (function): Hamiltonian potential
        potential_grad (function): Hamiltonian potential gradient
        mass_matrix (jax.Array): mass matrix
        momentum_noise (float): momentum noise
        integrator (object): integrator object
    -------------------------
    Returns:
        samples (jax.Array): samples
    """
    samples = []
    x = x_init
    # Initial momentum (gaussian), shape given by mass matrix
    p = jax.random.multivariate_normal(key, jnp.zeros(x.shape[0]), mass_matrix)
    for n in tqdm(range(n_samples + burn_in)):
        key, subkey = jax.random.split(key)
        # Sample noise vector
        mu = jax.random.multivariate_normal(subkey, jnp.zeros(x.shape[0]), mass_matrix)
        # Propose updated momentum and noise vector
        p_prop = jnp.sqrt(1 - momentum_noise) * p + jnp.sqrt(momentum_noise) * mu
        mu_prop = -jnp.sqrt(momentum_noise) * p + jnp.sqrt(1 - momentum_noise) * mu
        # Integrate Hamiltonian dynamics
        x_new, p_new = integrator.integrate(x, p_prop, potential_grad, n_steps, mass_matrix, step_size)
        # Computer enery error
        delta_H = Hamiltonian(x_new, p_new, potential, mass_matrix) - Hamiltonian(x, p_prop, potential, mass_matrix)
        if metropolize:
            # Metropolis-Hastings acceptance
            accept = jax.random.uniform(subkey) < jnp.exp(-delta_H)
            x, p = jax.lax.cond(accept, lambda _:(x_new, p_new), lambda _:(x, -p_prop), operand=None)
        else:
            x, p = x_new, p_new
        if n >= burn_in:
            samples.append(x)
    samples = jnp.stack(samples, axis = 0)
    return samples

def GHMC(x_init, potential_args, n_samples, burn_in, step_size, n_steps, 
        potential, mass_matrix, momentum_noise, integrator = VerletIntegrator(), n_chains = 4, RNG_key = 42, sampler = "GHMC"):
    """
    Multi-Chain Generalized Hamiltonian Monte-Carlo (GHMC) sampler.
    -------------------------
    Parameters:
        x_init (jax.Array): initial position
        potential_args (tuple): arguments for Hamiltonian potential
        n_samples (int): number of samples
        burn_in (int): burn-in samples
        step_size (float): step-size
        n_steps (int): number of integration steps
        potential (function): Hamiltonian potential
        mass_matrix (jax.Array): mass matrix
        momentum_noise (float): momentum noise
        integrator (object): integrator object
        n_chains (int): number of chains
        RNG_key (int): random number generator key
    -------------------------
    Returns:
        samples (jax.Array): samples
    """
    print(f"Running {sampler} sampler...")
    print("="*61)
    print(f"{'Num. Chains':^30}|{n_chains:^30}")
    print(f"{'Num. Samples':^30}|{n_samples:^30}")
    print(f"{'Num. Burn-In Iterations':^30}|{burn_in:^30}")
    print(f"{'Step-Size':^30}|{step_size:^30}")
    print(f"{'Num. Integration Steps':^30}|{n_steps:^30}")
    print(f"{'Momentum Noise':^30}|{momentum_noise:^30}")
    print("="*61)
    keys = jax.random.split(jax.random.PRNGKey(RNG_key), n_chains)
    x_init_repeated = jnp.repeat(x_init[None, :], n_chains, axis = 0)
    potential = jax.tree_util.Partial(potential, *potential_args)
    potential_grad = jax.grad(potential)
    vectorized_chain = jax.vmap(_single_chain_GHMC, in_axes=(0, None, None, None, None, None, None, None, None, None, 0))
    metropolize = (sampler != "MDMC" and sampler != "SLDMC")
    samples = vectorized_chain(x_init_repeated, n_samples, burn_in, step_size, n_steps, potential, potential_grad, mass_matrix, momentum_noise, integrator, keys, metropolize)
    return samples

def MALA(x_init, potential_args, n_samples, burn_in, step_size, 
        potential, mass_matrix, integrator = VerletIntegrator(), n_chains = 4, RNG_key = 42):
    """
    Multi-Chain Metropolis Adjusted Langevin Algorithm (MALA) sampler.
    -------------------------
    Parameters:
        x_init (jax.Array): initial position
        potential_args (tuple): arguments for Hamiltonian potential
        n_samples (int): number of samples
        burn_in (int): burn-in samples
        step_size (float): step-size
        potential (function): Hamiltonian potential
        mass_matrix (jax.Array): mass matrix
        integrator (object): integrator object
        n_chains (int): number of chains
        RNG_key (int): random number generator key
    -------------------------
    Returns:
        samples (jax.Array): samples
    """
    return GHMC(x_init = x_init, potential_args = potential_args, 
                n_samples = n_samples, burn_in = burn_in, step_size = step_size, 
                n_steps = 1, potential = potential, mass_matrix = mass_matrix, 
                momentum_noise = 1, integrator = integrator, n_chains = n_chains, RNG_key = RNG_key, sampler = "MALA")

def L2MC(x_init, potential_args, n_samples, burn_in, step_size, 
        potential, mass_matrix, momentum_noise, integrator = VerletIntegrator(), n_chains = 4, RNG_key = 42):
    """
    Multi-Chain Second-Order Langevin Monte Carlo (L2MC) sampler.
    -------------------------
    Parameters:
        x_init (jax.Array): initial position
        potential_args (tuple): arguments for Hamiltonian potential
        n_samples (int): number of samples
        burn_in (int): burn-in samples
        step_size (float): step-size
        potential (function): Hamiltonian potential
        mass_matrix (jax.Array): mass matrix
        momentum_noise (float): momentum noise
        integrator (object): integrator object
        n_chains (int): number of chains
        RNG_key (int): random number generator key
    -------------------------
    Returns:
        samples (jax.Array): samples
    """
    return GHMC(x_init = x_init, potential_args = potential_args, 
                n_samples = n_samples, burn_in = burn_in, step_size = step_size, 
                n_steps = 1, potential = potential, mass_matrix = mass_matrix, 
                momentum_noise = momentum_noise, integrator = integrator, n_chains = n_chains, RNG_key = RNG_key, sampler = "L2MC")

def MDMC(x_init, potential_args, n_samples, burn_in, step_size, n_steps,
        potential, mass_matrix, integrator = VerletIntegrator(), n_chains = 4, RNG_key = 42):
    """
    Multi-Chain Molecular Dynamics Monte Carlo (MDMC) sampler.
    -------------------------
    Parameters:
        x_init (jax.Array): initial position
        potential_args (tuple): arguments for Hamiltonian potential
        n_samples (int): number of samples
        burn_in (int): burn-in samples
        step_size (float): step-size
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
    return GHMC(x_init = x_init, potential_args = potential_args, 
                n_samples = n_samples, burn_in = burn_in, step_size = step_size, 
                n_steps = n_steps, potential = potential, mass_matrix = mass_matrix, 
                momentum_noise = 0, integrator = integrator, n_chains = n_chains, RNG_key = RNG_key, sampler = "MDMC")

def SLDMC(x_init, potential_args, n_samples, burn_in, step_size, n_steps,
        potential, mass_matrix, friction, integrator = VerletIntegrator(), n_chains = 4, RNG_key = 42):
    """
    Multi-Chain Stochastic Langevin Dynamics Monte Carlo (SLDMC) sampler.
    -------------------------
    Parameters:
        x_init (jax.Array): initial position
        potential_args (tuple): arguments for Hamiltonian potential
        n_samples (int): number of samples
        burn_in (int): burn-in samples
        step_size (float): step-size
        n_steps (int): number of integration steps
        potential (function): Hamiltonian potential
        mass_matrix (jax.Array): mass matrix
        friction (float): friction coefficient (gamma)
        integrator (object): integrator object
        n_chains (int): number of chains
        RNG_key (int): random number generator key
    -------------------------
    Returns:
        samples (jax.Array): samples
    """
    momentum_noise = jnp.sqrt(2 * friction * step_size)
    return GHMC(x_init = x_init, potential_args = potential_args, 
                n_samples = n_samples, burn_in = burn_in, step_size = step_size, 
                n_steps = n_steps, potential = potential, mass_matrix = mass_matrix, 
                momentum_noise = momentum_noise, integrator = integrator, n_chains = n_chains, 
                RNG_key = RNG_key, sampler = "SLDMC")

def MMHMC():
    # TODO: Implement MMHMC sampler
    raise NotImplementedError("MMHMC sampler not implemented yet!")

def RMHMC():
    # TODO: Implement RMHMC sampler
    raise NotImplementedError("RMHMC sampler not implemented yet!")
