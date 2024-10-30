import jax
import jax.numpy as jnp
from tqdm import tqdm
from functools import partial
from ..integrators import VerletIntegrator

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
    potential = jax.tree_util.Partial(potential, *potential_args)
    potential_grad = jax.grad(potential)
    vectorized_chain = jax.vmap(_single_chain_HMC, in_axes=(0, None, None, None, None, None, None, None, None, 0))
    samples = vectorized_chain(x_init_repeated, n_samples, burn_in, step_size, n_steps, potential, potential_grad, mass_matrix, integrator, keys)
    return samples

def _single_chain_GHMC(x_init, n_samples, burn_in, step_size, n_steps, 
        potential, potential_grad, mass_matrix, momentum_noise, integrator, key):
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
        # Metropolis-Hastings acceptance
        accept = jax.random.uniform(subkey) < jnp.exp(-delta_H)
        x, p = jax.lax.cond(accept, lambda _:(x_new, p_new), lambda _:(x, -p_prop), operand=None)
        if n >= burn_in:
            samples.append(x)
    samples = jnp.stack(samples, axis = 0)
    return samples

def GHMC(x_init, potential_args, n_samples, burn_in, step_size, n_steps, 
        potential, mass_matrix, momentum_noise, integrator = VerletIntegrator(), n_chains = 4, RNG_key = 42):
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
    print("Running GHMC sampler...")
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
    samples = vectorized_chain(x_init_repeated, n_samples, burn_in, step_size, n_steps, potential, potential_grad, mass_matrix, momentum_noise, integrator, keys)
    return samples

def MMHMC():
    # TODO: Implement MMHMC sampler
    raise NotImplementedError("MMHMC sampler not implemented yet!")


def _sAIA_HMC(x_init, n_samples, burn_in, step_size, n_steps, 
        potential, potential_grad, mass_matrix, integrator, key):
    """
    Single-Chain Hamiltonian Monte-Carlo (HMC) sampler (for s-AIA).
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
    frequencies = []
    acceptances = 0
    hessian_jit = jax.jit(jax.hessian(potential))
    x = x_init
    for n in range(n_samples + burn_in):
        key, subkey = jax.random.split(key)
        # Initial momentum (gaussian), shape given by mass matrix
        p = jax.random.multivariate_normal(subkey, jnp.zeros(x.shape[0]), mass_matrix)
        # Integrate Hamiltonian dynamics
        x_prop, p_prop = integrator.integrate(x, p, potential_grad, n_steps, mass_matrix, step_size)
        # Computer enery error
        delta_H = Hamiltonian(x_prop, p_prop, potential, mass_matrix) - Hamiltonian(x, p, potential, mass_matrix)
        # Metropolis-Hastings acceptance
        accept = jax.random.uniform(subkey) < jnp.exp(-delta_H)
        # If acceptance, add one to acceptances
        x = jax.lax.cond(accept, lambda _: x_prop, lambda _: x, operand=None)
        if n >= burn_in:
            samples.append(x)
            acceptances = jax.lax.cond(accept, lambda _: acceptances + 1, lambda _: acceptances, operand=None)
            # Compute Hessian of potential & frequencies (sqrt of eigenvalues)
            Hessian = hessian_jit(x)
            # FIXME: Something is going wrong here (eigenvalues are not being computed correctly)
            freqs_iter = jnp.sqrt(jnp.linalg.eigvals(Hessian))
            frequencies.append(freqs_iter)
    samples, frequencies = jnp.stack(samples, axis = 0), jnp.stack(frequencies, axis = 0)
    return samples, acceptances, frequencies

def _sAIA_Tuning(x_init, n_samples_tune, n_samples_check, step_size, n_steps, sensibility,
                              target_AR, potential, potential_grad, mass_matrix,
                              delta_step, integrator, key):
    tuned_step_size, N, N_tot = step_size, 0, 0
    while N_tot + n_samples_check < n_samples_tune:
        samples, N_acc, frequencies = _sAIA_HMC(x_init, n_samples = n_samples_check, burn_in = 0, step_size = tuned_step_size, 
                                         n_steps = n_steps, potential = potential, potential_grad = potential_grad,
                                         mass_matrix = mass_matrix, integrator = integrator, key = key)
        N += n_samples_check
        AR = N_acc / N
        if AR < target_AR - sensibility:
            tuned_step_size -= delta_step
            N = 0
        elif AR > target_AR + sensibility:
            tuned_step_size += delta_step
            N = 0
        N_tot += n_samples_check
    return tuned_step_size

def _sAIA_BurnIn(x_init, n_samples_burn_in, n_samples_prod, compute_freqs, step_size, n_steps, stage, potential, potential_grad, mass_matrix, integrator, key):
    samples, N_acc, frequencies = _sAIA_HMC(x_init, n_samples = n_samples_burn_in, burn_in = 0, step_size = step_size,
                                n_steps = n_steps, potential = potential, potential_grad = potential_grad,
                                mass_matrix = mass_matrix, integrator = integrator, key = key)
    frequencies = jnp.mean(frequencies, axis = 0)
    max_freq = jnp.max(frequencies)
    AR = N_acc / n_samples_burn_in
    dimensionless_step_sizes, step_sizes = None, None
    if compute_freqs:
        S = jnp.max(jnp.array([1, 2/(max_freq * step_size) * jnp.power((2*jnp.pi*(1 - AR)**2)/x_init.shape[0], 1/6)]))
        if S <= 2:
            fitting_factor = S
            stability_limit = 2*stage/(max_freq * fitting_factor)
            # Compute the n_samples_prod step-sizes by randomly sampling in the interval [0, stability_limit]
            step_sizes = jax.random.uniform(key, shape = (n_samples_prod, )) * stability_limit
            dimensionless_step_sizes = jax.lax.cond(S > 1, 
                                                    lambda _: (2*step_sizes/step_size)*jnp.power(2*jnp.pi*(1 - AR)**2/x_init.shape[0], 1/6),
                                                    lambda _: step_sizes * max_freq, 
                                                    operand = None)
    if not compute_freqs or S > 2:
        S_freq = jnp.max(jnp.array([1, 2/step_size * jnp.power((2*jnp.pi*(1 - AR)**2)/jnp.sum(frequencies**6), 1/6)]))
        std_dev_freq = jnp.std(frequencies)
        if std_dev_freq <= 1:
            fitting_factor = S_freq
            stability_limit = 2*stage/(max_freq * fitting_factor)
            step_sizes = jax.random.uniform(key, shape = (n_samples_prod, )) * stability_limit
            dimensionless_step_sizes = jax.lax.cond(S_freq > 1, 
                                                    lambda _: (2*max_freq*step_sizes/step_size)*jnp.power(2*jnp.pi*(1 - AR)**2/(jnp.sum(frequencies**6)), 1/6),
                                                    lambda _: step_sizes * max_freq, 
                                                    operand = None)
        elif std_dev_freq > 1:
            stability_limit = 2*stage/(S_freq * (max_freq - std_dev_freq))
            step_sizes = jax.random.uniform(key, shape = (n_samples_prod, )) * stability_limit
            dimensionless_step_sizes = jax.lax.cond(S_freq > 1, 
                                                    lambda _: (2*(max_freq - std_dev_freq)*step_sizes/step_size)*jnp.power(2*jnp.pi*(1 - AR)**2/(jnp.sum(frequencies**6)), 1/6),
                                                    lambda _: step_sizes * (max_freq - std_dev_freq), 
                                                    operand = None)
    return dimensionless_step_sizes, step_sizes

def sAIA(x_init, potential_args, n_samples_tune, n_samples_check, 
         n_samples_burn_in, n_samples_prod, potential, mass_matrix, 
         target_AR = 0.92, stage = 2, sensibility = 0.01, 
         delta_step = 0.01, compute_freqs = True, sampler = "HMC", RNG_key = 42):
    """
    s-AIA: Adaptive Integration Approach for Computation Statistics.

    Note: As of this version the s-AIA method is only supported for 2- & 3-stage
    Splitting Integrators w/ HMC sampling.
    TODO: Complete docstring
    -------------------------
    Parameters:
        x_init (jax.Array): initial position
        potential_args (tuple): arguments for Hamiltonian potential
        potential (function): Hamiltonian potential
        mass_matrix (jax.Array): mass matrix
        RNG_key (int): random number generator key
    -------------------------
    Returns:
        samples (jax.Array): samples
    """
    #TODO: Extend functionality to other samplers and generalize to k-stages
    if stage not in [2, 3]:
        raise NotImplementedError("Only 2- & 3-stage integrators are supported as of now.")
    if sampler not in ["HMC"]:
        raise NotImplementedError("Only HMC sampler is supported as of now.")
    print("Running s-AIA Adaptive Integration Scheme...")
    print("="*61)
    print(f"{'Sampler':^30}|{sampler:^30}")
    # TODO: Print other s-AIA parameters
    print("="*61)
    potential = jax.tree_util.Partial(potential, *potential_args)
    potential_grad = jax.grad(potential)
    # Step 1: Tuning Stage
    print("1) Tuning Stage...")
    n_samples, step_size, n_steps, integrator = n_samples_tune, 1/x_init.shape[0], 1, VerletIntegrator()
    print(f"\t- Number of Tuning Samples: {n_samples}")
    print(f"\t- Dimension of Data: {x_init.shape[0]}")
    print(f"\t- Initial Step-Size: {step_size}")
    tuned_step_size = _sAIA_Tuning(x_init, n_samples, n_samples_check, step_size, n_steps, 
                                   sensibility, target_AR, potential, potential_grad, mass_matrix, 
                                   delta_step, integrator, jax.random.PRNGKey(RNG_key))
    print(f"\t- Tuned Step-Size: {tuned_step_size}")
    print("="*61)
    # Step 2: Burn-In Stage
    print("2) Burn-In Stage...")
    n_samples, step_size = n_samples_burn_in, tuned_step_size
    print(f"\t- Number of Burn-In Samples: {n_samples}")
    dimensionless_step_sizes, step_sizes = _sAIA_BurnIn(x_init, n_samples, n_samples_prod, compute_freqs, step_size, n_steps, 
                                                        stage, potential, potential_grad, mass_matrix, integrator, 
                                                        jax.random.PRNGKey(RNG_key))
    print(f"\t- Dimensionless Step-Sizes: {dimensionless_step_sizes}")
    print(f"\t- Step-Sizes: {step_sizes}")
    print("="*61)
    # TODO: Continue here