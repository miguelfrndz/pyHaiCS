import jax
import jax.numpy as jnp
from tqdm import tqdm
from functools import partial
from ..integrators import VerletIntegrator, VV_2, ME_2, VV_3, ME_3, MSSI_2, MSSI_3, Integrator
from ..utils.metrics import acceptance_rate

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
    potential = jax.jit(jax.tree_util.Partial(potential, *potential_args))
    potential_grad = jax.jit(jax.grad(potential))
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

@jax.jit
def _compute_frequencies(Hessian):
    """
    Compute frequencies of a Hamiltonian system.
    -------------------------
    Parameters:
        potential_hessian (function): Hessian of Hamiltonian potential
        x (jax.Array): position
    -------------------------
    Returns:
        freqs (jax.Array): frequencies
    """
    freqs = jnp.sqrt(jnp.linalg.eigvals(Hessian))
    return freqs


def _sAIA_HMC(x_init, n_samples, burn_in, step_size, n_steps, 
    potential, potential_grad, potential_hessian, mass_matrix, integrator, key, phase_name):
    """
    Single-Chain Hamiltonian Monte-Carlo (HMC) sampler (for s-AIA).
    -------------------------
    Parameters:
    n_samples (int): number of samples
    burn_in (int): burn-in samples
    step_size (float or list): step size(s)
    n_steps (int or list): number of integration steps(s)
    potential (function): Hamiltonian potential
    potential_grad (function): Hamiltonian potential gradient
    mass_matrix (jax.Array): mass matrix
    integrator (object): integrator object
    -------------------------
    Returns:
    samples (jax.Array): samples
    """
    # Ensure step_size and n_steps are lists of the correct length
    if isinstance(step_size, (int, float)):
        step_size = [step_size] * n_samples
    if isinstance(n_steps, int):
        n_steps = [n_steps] * n_samples
    if isinstance(integrator, Integrator):
        integrator = [integrator] * n_samples
    assert len(step_size) == n_samples, "step_size must have length n_samples"
    assert len(n_steps) == n_samples, "n_steps must have length n_samples"
    assert len(integrator) == n_samples, "integrator must have length n_samples"

    samples = []
    frequencies = []
    acceptances = 0
    x = x_init
    for n in tqdm(range(n_samples + burn_in), desc = f"\t- Running {phase_name} Phase HMC", ncols = 100):
        key, subkey = jax.random.split(key)
        # Initial momentum (gaussian), shape given by mass matrix
        p = jax.random.multivariate_normal(subkey, jnp.zeros(x.shape[0]), mass_matrix)
        # Integrate Hamiltonian dynamics
        current_step_size = step_size[min(n - burn_in, n_samples - 1)] if n >= burn_in else step_size[0]
        current_n_steps = n_steps[min(n - burn_in, n_samples - 1)] if n >= burn_in else n_steps[0]
        current_integrator = integrator[min(n - burn_in, n_samples - 1)] if n >= burn_in else integrator[0]
        x_prop, p_prop = current_integrator.integrate(x, p, potential_grad, current_n_steps, mass_matrix, current_step_size)
        # Compute energy error
        delta_H = Hamiltonian(x_prop, p_prop, potential, mass_matrix) - Hamiltonian(x, p, potential, mass_matrix)
        # Metropolis-Hastings acceptance
        accept = jax.random.uniform(subkey) < jnp.exp(-delta_H)
        # If acceptance, add one to acceptances
        x = jax.lax.cond(accept, lambda _: x_prop, lambda _: x, operand = None)
        if n >= burn_in:
            samples.append(x)
            acceptances = jax.lax.cond(accept, lambda _: acceptances + 1, lambda _: acceptances, operand = None)
            # Compute Hessian of potential & frequencies (sqrt of eigenvalues)
            Hessian = potential_hessian(x)
            freqs_iter = _compute_frequencies(Hessian)
            frequencies.append(freqs_iter)
    samples, frequencies = jnp.stack(samples, axis = 0), jnp.stack(frequencies, axis = 0)
    return samples, acceptances, frequencies

def _sAIA_GHMC(x_init, n_samples, burn_in, step_size, n_steps, 
    potential, potential_grad, potential_hessian, mass_matrix, momentum_noise_lower, momentum_noise_upper, integrator, key, phase_name):
    """
    Single-Chain Generalized Hamiltonian Monte-Carlo (GHMC) sampler (for s-AIA).
    -------------------------
    Parameters:
    n_samples (int): number of samples
    burn_in (int): burn-in samples
    step_size (float or list): step size(s)
    n_steps (int or list): number of integration steps(s)
    potential (function): Hamiltonian potential
    potential_grad (function): Hamiltonian potential gradient
    mass_matrix (jax.Array): mass matrix
    momentum_noise (float): momentum noise
    integrator (object): integrator object
    -------------------------
    Returns:
    samples (jax.Array): samples
    """
    # Ensure step_size and n_steps are lists of the correct length
    if isinstance(step_size, (int, float)):
        step_size = [step_size] * n_samples
    if isinstance(n_steps, int):
        n_steps = [n_steps] * n_samples
    if isinstance(integrator, Integrator):
        integrator = [integrator] * n_samples
    # Sample momentum noise from uniform distribution in [momentum_noise_lower, momentum_noise_upper]
    momentum_noise = jax.random.uniform(key, shape = (n_samples, )) * (momentum_noise_upper - momentum_noise_lower) + momentum_noise_lower
    assert len(step_size) == n_samples, "step_size must have length n_samples"
    assert len(n_steps) == n_samples, "n_steps must have length n_samples"
    assert len(integrator) == n_samples, "integrator must have length n_samples"
    assert len(momentum_noise) == n_samples, "momentum_noise must have length n_samples"

    samples = []
    frequencies = []
    acceptances = 0
    x = x_init
    # Initial momentum (gaussian), shape given by mass matrix
    p = jax.random.multivariate_normal(key, jnp.zeros(x.shape[0]), mass_matrix)
    for n in tqdm(range(n_samples + burn_in), desc=f"\t- Running {phase_name} Phase GHMC", ncols=100):
        key, subkey = jax.random.split(key)
        # Sample noise vector
        mu = jax.random.multivariate_normal(subkey, jnp.zeros(x.shape[0]), mass_matrix)
        # Propose updated momentum and noise vector
        current_momentum_noise = momentum_noise[min(n - burn_in, n_samples - 1)] if n >= burn_in else momentum_noise[0]
        p_prop = jnp.sqrt(1 - current_momentum_noise) * p + jnp.sqrt(current_momentum_noise) * mu
        mu_prop = -jnp.sqrt(current_momentum_noise) * p + jnp.sqrt(1 - current_momentum_noise) * mu
        # Integrate Hamiltonian dynamics
        current_step_size = step_size[min(n - burn_in, n_samples - 1)] if n >= burn_in else step_size[0]
        current_n_steps = n_steps[min(n - burn_in, n_samples - 1)] if n >= burn_in else n_steps[0]
        current_integrator = integrator[min(n - burn_in, n_samples - 1)] if n >= burn_in else integrator[0]
        x_new, p_new = current_integrator.integrate(x, p_prop, potential_grad, current_n_steps, mass_matrix, current_step_size)
        # Compute energy error
        delta_H = Hamiltonian(x_new, p_new, potential, mass_matrix) - Hamiltonian(x, p_prop, potential, mass_matrix)
        # Metropolis-Hastings acceptance
        accept = jax.random.uniform(subkey) < jnp.exp(-delta_H)
        # If acceptance, add one to acceptances
        x, p = jax.lax.cond(accept, lambda _: (x_new, p_new), lambda _: (x, -p_prop), operand=None)
        if n >= burn_in:
            samples.append(x)
            acceptances = jax.lax.cond(accept, lambda _: acceptances + 1, lambda _: acceptances, operand=None)
            # Compute Hessian of potential & frequencies (sqrt of eigenvalues)
            Hessian = potential_hessian(x)
            freqs_iter = _compute_frequencies(Hessian)
            frequencies.append(freqs_iter)
    samples, frequencies = jnp.stack(samples, axis=0), jnp.stack(frequencies, axis=0)
    return samples, acceptances, frequencies

def _sAIA_Tuning(x_init, n_samples_tune, n_samples_check, step_size, n_steps, sensibility,
                              target_AR, potential, potential_grad, potential_hessian, mass_matrix,
                              delta_step, integrator, sampler, momentum_noise_lower, momentum_noise_upper, key):
    tuned_step_size, N, N_tot = step_size, 0, 0
    while N_tot + n_samples_check < n_samples_tune:
        if sampler == "HMC":
            samples, N_acc, frequencies = _sAIA_HMC(x_init, n_samples = n_samples_check, burn_in = 0, step_size = tuned_step_size, 
                                         n_steps = n_steps, potential = potential, potential_grad = potential_grad,
                                         potential_hessian = potential_hessian, mass_matrix = mass_matrix, integrator = integrator, key = key,
                                         phase_name = "Tuning")
        elif sampler == "GHMC":
            samples, N_acc, frequencies = _sAIA_GHMC(x_init, n_samples = n_samples_check, burn_in = 0, step_size = tuned_step_size, 
                                         n_steps = n_steps, potential = potential, potential_grad = potential_grad,
                                         potential_hessian = potential_hessian, mass_matrix = mass_matrix, 
                                         momentum_noise_lower = momentum_noise_lower, momentum_noise_upper = momentum_noise_upper,
                                         integrator = integrator, key = key, phase_name = "Tuning")
        N += n_samples_check
        AR = acceptance_rate(N_acc, n_samples_check)
        if AR < target_AR - sensibility:
            tuned_step_size -= delta_step
            N = 0
        elif AR > target_AR + sensibility:
            tuned_step_size += delta_step
            N = 0
        N_tot += n_samples_check
    return tuned_step_size

def _sAIA_BurnIn(x_init, n_samples_burn_in, n_samples_prod, compute_freqs, step_size, 
                 n_steps, stage, potential, potential_grad, potential_hessian, 
                 mass_matrix, integrator, sampler, momentum_noise_lower,
                 momentum_noise_upper, key):
    if sampler == "HMC":
        samples, N_acc, frequencies = _sAIA_HMC(x_init, n_samples = n_samples_burn_in, burn_in = 0, step_size = step_size,
                                    n_steps = n_steps, potential = potential, potential_grad = potential_grad, 
                                    potential_hessian = potential_hessian, mass_matrix = mass_matrix, 
                                    integrator = integrator, key = key, phase_name = "Burn-In")
    elif sampler == "GHMC":
        samples, N_acc, frequencies = _sAIA_GHMC(x_init, n_samples = n_samples_burn_in, burn_in = 0, step_size = step_size, 
                                    n_steps = n_steps, potential = potential, potential_grad = potential_grad, 
                                    potential_hessian = potential_hessian, mass_matrix = mass_matrix, 
                                    momentum_noise_lower = momentum_noise_lower, momentum_noise_upper = momentum_noise_upper,
                                    integrator = integrator, key = key, phase_name = "Burn-In")
    frequencies = jnp.mean(frequencies, axis = 0)
    # Handle complex frequencies by taking the absolute value
    frequencies = jnp.abs(frequencies)
    max_freq = jnp.max(frequencies)
    AR = acceptance_rate(N_acc, n_samples_burn_in)
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

@jax.jit
def _rho_2(step_size, b):
    numerator = step_size**4 * (2 * b**2 * (1/2 - b) * step_size**2 + 4 * b**2 - 6 * b + 1)**2
    denominator = 8 * (2 - b * step_size**2) * (2 - (1/2 - b) * step_size**2) * (1 - b * (1/2 - b) * step_size**2)
    return numerator / denominator

@jax.jit
def _rho_3(step_size, b):
    numerator = step_size**4 * (-3 * b**4 + 8 * b**3 - 19/4 * b**2 + b + b**2 * step_size**2 * (b**3 - 5/4 * b**2 + b/2 - 1/16) - 1/16)**2
    denominator = 2 * (3 * b - b * step_size**2 * (b - 1/4) - 1) * (1 - 3 * b - b * step_size**2 * (b - 1/2)**2) * (-9 * b**2 + 6 * b - step_size**2 * (b**3 - 5/4 * b**2 + b/2 - 1/16) - 1)
    return numerator / denominator

def _sAIA_OptimalCoeffs(dimensionless_step_sizes, stage, key, n_coeff_samples = 20):
    rho, b_MEk, b_VVk = None, None, None
    if stage == 2:
        rho = _rho_2
        b_MEk = ME_2().b
        b_VVk = VV_2().b
    elif stage == 3:
        rho = _rho_3
        b_MEk = ME_3().b
        b_VVk = VV_3().b
    else:
        raise NotImplementedError("Only 2- & 3-stage integrators are supported as of now.")
    optimal_coeffs = []
    # Sample b values between b_MEk and b_VVk
    for i in range(dimensionless_step_sizes.shape[0]):
        b_values = jax.random.uniform(jax.random.PRNGKey(key), shape = (n_coeff_samples, )) * (b_VVk - b_MEk) + b_MEk
        step_sizes = jax.random.uniform(jax.random.PRNGKey(key), shape = (n_coeff_samples, )) * dimensionless_step_sizes[i]
        max_rho = []
        for b in b_values:
            rho_vals = jax.vmap(rho, in_axes = (0, None))(step_sizes, b)
            max_rho.append(jnp.max(rho_vals))
        max_rho = jnp.array(max_rho)
        optimal_b = b_values[jnp.argmin(max_rho)]
        optimal_coeffs.append(optimal_b)
    optimal_coeffs = jnp.array(optimal_coeffs)
    return optimal_coeffs

def lambda_phi(stage, a = None, b = None):
    if stage == 2:
        lambda_2 = (6*b - 1)/24
        return lambda_2
    elif stage == 3:
        lambda_3 = (1 - 6*a * (1 - a) * (1 - 2*b))/12
        return lambda_3

def optimal_momentum_noise(step_size_nondim, stage, D, a = None, b = None):
    lambda_phi_val = lambda_phi(stage, a, b)
    phi_opt = jnp.minimum(1, -jnp.log(0.999)/D * (1 + 2 * step_size_nondim ** 2 * lambda_phi_val)/(2 * step_size_nondim**4 * lambda_phi_val ** 2))
    return phi_opt

def sAIA(x_init, potential_args, n_samples_tune, n_samples_check, 
         n_samples_burn_in, n_samples_prod, potential, mass_matrix, 
         target_AR = 0.92, stage = 2, sensibility = 0.01, 
         delta_step = 0.01, compute_freqs = True, sampler = "HMC", RNG_key = 42):
    """
    s-AIA: Adaptive Integration Approach for Computation Statistics.

    Note: As of this version the s-AIA method is only supported for 2- & 3-stage
    Splitting Integrators w/ HMC, GHMC sampling.
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
    if sampler not in ["HMC", "GHMC"]:
        raise NotImplementedError("Only HMC & GHMC samplers are supported as of now.")
    print("Running s-AIA Adaptive Integration Scheme...")
    print("="*61)
    print(f"{'Sampler':^30}|{sampler:^30}")
    print(f"{'Num. Samples Tune':^30}|{n_samples_tune:^30}")
    print(f"{'Num. Samples Check':^30}|{n_samples_check:^30}")
    print(f"{'Num. Samples Burn-In':^30}|{n_samples_burn_in:^30}")
    print(f"{'Num. Samples Prod':^30}|{n_samples_prod:^30}")
    print(f"{'Stage':^30}|{stage:^30}")
    print(f"{'Target AR':^30}|{target_AR:^30}")
    print(f"{'Sensibility':^30}|{sensibility:^30}")
    print(f"{'Delta Step':^30}|{delta_step:^30}")
    print(f"{'Compute Freqs':^30}|{('Yes' if compute_freqs else 'No'):^30}")
    print("="*61)
    potential = jax.jit(jax.tree_util.Partial(potential, *potential_args))
    potential_grad = jax.jit(jax.grad(potential))
    potential_hessian = jax.jit(jax.hessian(potential))
    # Step 1: Tuning Stage
    print("1) Tuning Stage...")
    n_samples, step_size, n_steps, integrator = n_samples_tune, 1/x_init.shape[0], 1, VerletIntegrator()
    momentum_noise_lower, momentum_noise_upper = None, None
    if sampler == "GHMC":
        if stage == 2: a, b = 0, 1/4
        elif stage == 3: a, b = 1/3, 1/6
        momentum_noise_lower = optimal_momentum_noise(2.0772, stage, x_init.shape[0], a, b)
        momentum_noise_upper = optimal_momentum_noise(stage, stage, x_init.shape[0], a, b)
    print(f"\t- Number of Tuning Samples: {n_samples}")
    print(f"\t- Dimension of Data: {x_init.shape[0]}")
    print(f"\t- Initial Step-Size: {step_size}")
    if sampler == "GHMC":
        print(f"\t- Initial Momentum Noise (Lower Bound): {momentum_noise_lower}")
        print(f"\t- Initial Momentum Noise (Upper Bound): {momentum_noise_upper}")
    tuned_step_size = _sAIA_Tuning(x_init, n_samples, n_samples_check, step_size, n_steps, 
                                   sensibility, target_AR, potential, potential_grad, potential_hessian,
                                   mass_matrix, delta_step, integrator, sampler, momentum_noise_lower, momentum_noise_upper, jax.random.PRNGKey(RNG_key))
    print(f"\t- Tuned Step-Size: {tuned_step_size}")
    print("="*61)
    # Step 2: Burn-In Stage
    print("2) Burn-In Stage...")
    n_samples, step_size = n_samples_burn_in, tuned_step_size
    print(f"\t- Number of Burn-In Samples: {n_samples}")
    dimensionless_step_sizes, step_sizes = _sAIA_BurnIn(x_init, n_samples, n_samples_prod, compute_freqs, step_size, n_steps, 
                                                        stage, potential, potential_grad, potential_hessian, mass_matrix, 
                                                        integrator, sampler, momentum_noise_lower, 
                                                        momentum_noise_upper, jax.random.PRNGKey(RNG_key))
    print(f"\t- Dimensionless Step-Sizes: {dimensionless_step_sizes}")
    print(f"\t- Step-Sizes: {step_sizes}")
    opt_integration_coeffs = _sAIA_OptimalCoeffs(dimensionless_step_sizes, stage, RNG_key)
    print(f"\t- Optimal Integration Coefficients: {opt_integration_coeffs}")
    print("="*61)
    # Step 3: Production Stage
    print("3) Production Stage...")
    n_steps = jax.random.randint(jax.random.PRNGKey(RNG_key), shape=(n_samples_prod,), minval=1, maxval=2 * (x_init.shape[0] / step_sizes) - 1)
    print(f"\t- Number of Steps: {n_steps}")
    if stage == 2:
        integrator = [MSSI_2(b) for b in opt_integration_coeffs]
    elif stage == 3:
        a_coeffs = [(1 + 2*b)/(2*(6*b-2)) for b in opt_integration_coeffs]
        integrator = [MSSI_3(a, b) for a, b in zip(a_coeffs, opt_integration_coeffs)]
    assert len(integrator) == n_samples_prod, "Number of integrators must be equal to number of samples"
    if sampler == "HMC":
        samples, _, _ = _sAIA_HMC(x_init, n_samples = n_samples_prod, burn_in = 100, step_size = step_sizes, 
                                n_steps = n_steps, potential = potential, potential_grad = potential_grad, 
                                potential_hessian = potential_hessian, mass_matrix = mass_matrix, 
                                integrator = integrator, key = jax.random.PRNGKey(RNG_key), phase_name = "Production")
    elif sampler == "GHMC":
        samples, _, _ = _sAIA_GHMC(x_init, n_samples = n_samples_prod, burn_in = 100, step_size = step_sizes, 
                                n_steps = n_steps, potential = potential, potential_grad = potential_grad, 
                                potential_hessian = potential_hessian, mass_matrix = mass_matrix, 
                                momentum_noise_lower = momentum_noise_lower, momentum_noise_upper = momentum_noise_upper,
                                integrator = integrator, key = jax.random.PRNGKey(RNG_key), phase_name = "Production")
    print("="*61)
    return samples
