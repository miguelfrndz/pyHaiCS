import jax
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm
from functools import partial
from ..utils.hamiltonian import Hamiltonian, Modified_Hamiltonian, Extended_Hamiltonian
from ..integrators.integrators import VerletIntegrator, VV_2, ME_2, VV_3, ME_3, MSSI_2, MSSI_3, Integrator, get_mhmc_coeffs
from ..utils.metrics import acceptance_rate

@jax.jit
def _compute_frequencies(Hessian):
    """
    Compute frequencies of a Hamiltonian system.
    -------------------------
    Parameters:
        Hessian (jax.Array): Hessian matrix
    -------------------------
    Returns:
        freqs (jax.Array): frequencies
    """
    Hessian = 0.5 * (Hessian + Hessian.T)
    eigvals = jnp.linalg.eigvalsh(Hessian)
    return jnp.sqrt(jnp.clip(eigvals, a_min = 0.0))


def _expand_schedule(values, n_samples, name):
    if isinstance(values, Integrator):
        values = [values] * n_samples
    elif isinstance(values, (int, float)):
        values = [values] * n_samples
    elif isinstance(values, jax.Array) and values.ndim == 0:
        values = [values] * n_samples
    else:
        values = list(values)

    if len(values) != n_samples:
        raise ValueError(f"{name} must have length {n_samples}")
    return values


@partial(jax.jit, static_argnums=(2,))
def _integrate_mssi_2(x, p, potential_grad, n_steps, mass_matrix, step_size, b):
    def body_fun(_, state):
        x, p = state
        p = p - step_size * b * potential_grad(x)
        x = x + (step_size / 2) * jnp.linalg.solve(mass_matrix, p)
        p = p - step_size * (1 - 2 * b) * potential_grad(x)
        x = x + (step_size / 2) * jnp.linalg.solve(mass_matrix, p)
        p = p - step_size * b * potential_grad(x)
        return x, p

    return jax.lax.fori_loop(0, n_steps, body_fun, (x, p))


@partial(jax.jit, static_argnums=(2,))
def _integrate_mssi_3(x, p, potential_grad, n_steps, mass_matrix, step_size, a, b):
    def body_fun(_, state):
        x, p = state
        p = p - step_size * b * potential_grad(x)
        x = x + step_size * a * jnp.linalg.solve(mass_matrix, p)
        p = p - step_size * (0.5 - b) * potential_grad(x)
        x = x + step_size * (1 - 2 * a) * jnp.linalg.solve(mass_matrix, p)
        p = p - step_size * (0.5 - b) * potential_grad(x)
        x = x + step_size * a * jnp.linalg.solve(mass_matrix, p)
        p = p - step_size * b * potential_grad(x)
        return x, p

    return jax.lax.fori_loop(0, n_steps, body_fun, (x, p))


def _integrate_adaptive_step(x, p, potential_grad, n_steps, mass_matrix, step_size, integrator_spec):
    if isinstance(integrator_spec, Integrator):
        return integrator_spec.integrate(x, p, potential_grad, n_steps, mass_matrix, step_size)

    stage = integrator_spec[0]
    if stage == 2:
        _, b = integrator_spec
        return _integrate_mssi_2(x, p, potential_grad, n_steps, mass_matrix, step_size, b)
    if stage == 3:
        _, a, b = integrator_spec
        return _integrate_mssi_3(x, p, potential_grad, n_steps, mass_matrix, step_size, a, b)
    raise NotImplementedError(f"Unsupported adaptive integrator stage: {stage}")


def _sAIA_HMC(x_init, n_samples, burn_in, step_size, n_steps, 
    potential, potential_grad, potential_hessian, mass_matrix, integrator, key, phase_name,
    return_state = False):
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
    potential_hessian (function): Hamiltonian potential Hessian
    mass_matrix (jax.Array): mass matrix
    integrator (object): integrator object
    -------------------------
    Returns:
    samples (jax.Array): samples
    """
    step_size = _expand_schedule(step_size, n_samples, "step_size")
    n_steps = _expand_schedule(n_steps, n_samples, "n_steps")
    integrator = _expand_schedule(integrator, n_samples, "integrator")

    samples = []
    frequencies = []
    acceptances = 0
    x = x_init
    for n in tqdm(range(n_samples + burn_in), desc = f"\t- Running {phase_name} Phase HMC", ncols = 100):
        key, momentum_key, accept_key = jax.random.split(key, 3)
        # Initial momentum (gaussian), shape given by mass matrix
        p = jax.random.multivariate_normal(momentum_key, jnp.zeros(x.shape[0]), mass_matrix)
        # Integrate Hamiltonian dynamics
        current_idx = min(n - burn_in, n_samples - 1) if n >= burn_in else 0
        current_step_size = step_size[current_idx]
        current_n_steps = int(n_steps[current_idx])
        current_integrator = integrator[current_idx]
        x_prop, p_prop = _integrate_adaptive_step(x, p, potential_grad, current_n_steps, mass_matrix, current_step_size, current_integrator)
        # Compute energy error
        delta_H = Hamiltonian(x_prop, p_prop, potential, mass_matrix) - Hamiltonian(x, p, potential, mass_matrix)
        # Metropolis-Hastings acceptance
        accept = jax.random.uniform(accept_key) < jnp.exp(-delta_H)
        # If acceptance, add one to acceptances
        x = jax.lax.cond(accept, lambda _: x_prop, lambda _: x, operand = None)
        if n >= burn_in:
            samples.append(x)
            acceptances += int(accept)
            if potential_hessian is not None:
                Hessian = potential_hessian(x)
                freqs_iter = _compute_frequencies(Hessian)
            else:
                freqs_iter = jnp.ones(x.shape[0])
            frequencies.append(freqs_iter)
    samples = jnp.stack(samples, axis = 0)
    frequencies = jnp.stack(frequencies, axis = 0)
    if return_state:
        return samples, acceptances, frequencies, x, None, key
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
    potential_hessian (function): Hamiltonian potential Hessian
    mass_matrix (jax.Array): mass matrix
    momentum_noise_lower (float): lower bound for momentum noise
    momentum_noise_upper (float): upper bound for momentum noise
    integrator (object): integrator object
    -------------------------
    Returns:
    samples (jax.Array): samples
    """
    return _sAIA_GHMC_stateful(
        x_init, None, n_samples, burn_in, step_size, n_steps, potential, potential_grad,
        potential_hessian, mass_matrix, momentum_noise_lower, momentum_noise_upper,
        integrator, key, phase_name, return_state = False
    )


def _sAIA_GHMC_stateful(x_init, p_init, n_samples, burn_in, step_size, n_steps,
    potential, potential_grad, potential_hessian, mass_matrix, momentum_noise_lower,
    momentum_noise_upper, integrator, key, phase_name, return_state = False):
    step_size = _expand_schedule(step_size, n_samples, "step_size")
    n_steps = _expand_schedule(n_steps, n_samples, "n_steps")
    integrator = _expand_schedule(integrator, n_samples, "integrator")
    noise_lower = jnp.minimum(momentum_noise_lower, momentum_noise_upper)
    noise_upper = jnp.maximum(momentum_noise_lower, momentum_noise_upper)
    key, momentum_noise_key, momentum_key = jax.random.split(key, 3)
    momentum_noise = jax.random.uniform(momentum_noise_key, shape = (n_samples, )) * (noise_upper - noise_lower) + noise_lower

    samples = []
    frequencies = []
    acceptances = 0
    x = x_init
    p = p_init if p_init is not None else jax.random.multivariate_normal(momentum_key, jnp.zeros(x.shape[0]), mass_matrix)
    for n in tqdm(range(n_samples + burn_in), desc=f"\t- Running {phase_name} Phase GHMC", ncols=100):
        key, noise_key, accept_key = jax.random.split(key, 3)
        # Sample noise vector
        mu = jax.random.multivariate_normal(noise_key, jnp.zeros(x.shape[0]), mass_matrix)
        # Propose updated momentum and noise vector
        current_idx = min(n - burn_in, n_samples - 1) if n >= burn_in else 0
        current_momentum_noise = momentum_noise[current_idx]
        p_prop = jnp.sqrt(1 - current_momentum_noise) * p + jnp.sqrt(current_momentum_noise) * mu
        mu_prop = -jnp.sqrt(current_momentum_noise) * p + jnp.sqrt(1 - current_momentum_noise) * mu
        # Integrate Hamiltonian dynamics
        current_step_size = step_size[current_idx]
        current_n_steps = int(n_steps[current_idx])
        current_integrator = integrator[current_idx]
        x_new, p_new = _integrate_adaptive_step(x, p_prop, potential_grad, current_n_steps, mass_matrix, current_step_size, current_integrator)
        # Compute energy error
        delta_H = Hamiltonian(x_new, p_new, potential, mass_matrix) - Hamiltonian(x, p_prop, potential, mass_matrix)
        # Metropolis-Hastings acceptance
        accept = jax.random.uniform(accept_key) < jnp.exp(-delta_H)
        # If acceptance, add one to acceptances
        x, p = jax.lax.cond(accept, lambda _: (x_new, p_new), lambda _: (x, -p_prop), operand=None)
        if n >= burn_in:
            samples.append(x)
            acceptances += int(accept)
            if potential_hessian is not None:
                Hessian = potential_hessian(x)
                freqs_iter = _compute_frequencies(Hessian)
            else:
                freqs_iter = jnp.ones(x.shape[0])
            frequencies.append(freqs_iter)
    samples = jnp.stack(samples, axis=0)
    frequencies = jnp.stack(frequencies, axis=0)
    if return_state:
        return samples, acceptances, frequencies, x, p, key
    return samples, acceptances, frequencies

def _sAIA_Tuning(x_init, n_samples_tune, n_samples_check, step_size, n_steps, sensibility,
                              target_AR, potential, potential_grad, potential_hessian, mass_matrix,
                              delta_step, integrator, sampler, momentum_noise_lower, momentum_noise_upper, key):
    """
    Tuning stage for the s-AIA method
    -------------------------
    Parameters:
    n_samples_tune (int): number of samples for tuning
    n_samples_check (int): number of samples for checking acceptance rate
    step_size (float): initial step size
    n_steps (int): number of integration steps
    sensibility (float): sensibility for acceptance rate
    target_AR (float): target acceptance rate
    potential (function): Hamiltonian potential
    potential_grad (function): Hamiltonian potential gradient
    potential_hessian (function): Hamiltonian potential Hessian
    mass_matrix (jax.Array): mass matrix
    delta_step (float): step size increment/decrement
    integrator (object): integrator object
    sampler (str): sampler type
    momentum_noise_lower (float): lower bound for momentum noise
    momentum_noise_upper (float): upper bound for momentum noise
    key (int): random number generator key
    -------------------------
    Returns:
    tuned_step_size (float): tuned step size
    """
    tuned_step_size, N, N_acc_window, N_tot = step_size, 0, 0, 0
    AR = 0.0
    x = x_init
    p = None
    while N_tot + n_samples_check < n_samples_tune:
        if sampler == "HMC":
            _, N_acc, _, x, _, key = _sAIA_HMC(
                x, n_samples = n_samples_check, burn_in = 0, step_size = tuned_step_size,
                n_steps = n_steps, potential = potential, potential_grad = potential_grad,
                potential_hessian = potential_hessian, mass_matrix = mass_matrix, integrator = integrator,
                key = key, phase_name = "Tuning", return_state = True
            )
        elif sampler == "GHMC":
            _, N_acc, _, x, p, key = _sAIA_GHMC_stateful(
                x, p, n_samples = n_samples_check, burn_in = 0, step_size = tuned_step_size,
                n_steps = n_steps, potential = potential, potential_grad = potential_grad,
                potential_hessian = potential_hessian, mass_matrix = mass_matrix,
                momentum_noise_lower = momentum_noise_lower, momentum_noise_upper = momentum_noise_upper,
                integrator = integrator, key = key, phase_name = "Tuning", return_state = True
            )
        N += n_samples_check
        N_acc_window += N_acc
        AR = acceptance_rate(N_acc_window, N)
        if AR < target_AR - sensibility:
            tuned_step_size -= delta_step
            N, N_acc_window = 0, 0
        elif AR > target_AR + sensibility:
            tuned_step_size += delta_step
            N, N_acc_window = 0, 0
        N_tot += n_samples_check
    return tuned_step_size, AR, x, p, key

def _sAIA_BurnIn(x_init, n_samples_burn_in, n_samples_prod, compute_freqs, step_size, 
                 n_steps, stage, potential, potential_grad, potential_hessian, 
                 mass_matrix, integrator, sampler, momentum_noise_lower,
                 momentum_noise_upper, key, p_init = None):
    """
    Burn-In stage for the s-AIA method
    -------------------------
    Parameters:
    n_samples_burn_in (int): number of samples for burn-in
    n_samples_prod (int): number of samples for production
    compute_freqs (bool): compute frequencies
    step_size (float): step size
    n_steps (int): number of integration steps
    stage (int): number of stages
    potential (function): Hamiltonian potential
    potential_grad (function): Hamiltonian potential gradient
    potential_hessian (function): Hamiltonian potential Hessian
    mass_matrix (jax.Array): mass matrix
    integrator (object): integrator object
    sampler (str): sampler type
    momentum_noise_lower (float): lower bound for momentum noise
    momentum_noise_upper (float): upper bound for momentum noise
    key (int): random number generator key
    -------------------------
    Returns:
    dimensionless_step_sizes (jax.Array): dimensionless step sizes
    step_sizes (jax.Array): step sizes
    """
    if sampler == "HMC":
        samples, N_acc, frequencies, x_final, p_final, key = _sAIA_HMC(
            x_init, n_samples = n_samples_burn_in, burn_in = 0, step_size = step_size,
            n_steps = n_steps, potential = potential, potential_grad = potential_grad,
            potential_hessian = potential_hessian, mass_matrix = mass_matrix,
            integrator = integrator, key = key, phase_name = "Burn-In", return_state = True
        )
    elif sampler == "GHMC":
        samples, N_acc, frequencies, x_final, p_final, key = _sAIA_GHMC_stateful(
            x_init, p_init, n_samples = n_samples_burn_in, burn_in = 0, step_size = step_size,
            n_steps = n_steps, potential = potential, potential_grad = potential_grad,
            potential_hessian = potential_hessian, mass_matrix = mass_matrix,
            momentum_noise_lower = momentum_noise_lower, momentum_noise_upper = momentum_noise_upper,
            integrator = integrator, key = key, phase_name = "Burn-In", return_state = True
        )
    frequencies = jnp.mean(frequencies, axis = 0)
    max_freq = jnp.max(frequencies)
    AR = acceptance_rate(N_acc, n_samples_burn_in)
    dimensionless_step_sizes, step_sizes = None, None
    if potential_hessian is None:
        S = jnp.max(jnp.array([1, 2/(step_size) * jnp.power((2*jnp.pi*(1 - AR)**2)/x_init.shape[0], 1/6)]))
        fitting_factor = S
        t_ColSI = stage/(fitting_factor)
        # Compute the n_samples_prod step-sizes by randomly sampling in the interval [h_lower, h_ColSI]
        if stage == 3:  t_lower = 2.0772/(fitting_factor)
        elif stage == 2: t_lower = 1.5/(fitting_factor)
        else: raise NotImplementedError("Only 2- & 3-stage integrators are supported as of now.")
        step_sizes = jax.random.uniform(key, shape = (n_samples_prod, )) * (t_ColSI - t_lower) + t_lower 
        dimensionless_step_sizes = jax.lax.cond(S > 1, 
                                                lambda _: (2*step_sizes/step_size)*jnp.power(2*jnp.pi*(1 - AR)**2/x_init.shape[0], 1/6),
                                                lambda _: step_sizes * max_freq, 
                                                operand = None)
    else:
        if stage == 3:  h_lower = 2.0772
        elif stage == 2: h_lower = 1.5 # Arbitrary for 2-stage, should be changed
        else: raise NotImplementedError("Only 2- & 3-stage integrators are supported as of now.")
        
        if compute_freqs:
            S = jnp.max(jnp.array([1, 2/(max_freq * step_size) * jnp.power((2*jnp.pi*(1 - AR)**2)/x_init.shape[0], 1/6)]))
            if S <= 2:
                fitting_factor = S
                t_ColSI = stage/(max_freq * fitting_factor)
                t_lower = h_lower/(max_freq * fitting_factor)
                # stability_limit = 2*stage/(max_freq * fitting_factor)
                # Compute the n_samples_prod step-sizes by randomly sampling in the interval [h_lower, h_ColSI]
                step_sizes = jax.random.uniform(key, shape = (n_samples_prod, )) * (t_ColSI - t_lower) + t_lower
                dimensionless_step_sizes = jax.lax.cond(S > 1, 
                                                        lambda _: (2*step_sizes/step_size)*jnp.power(2*jnp.pi*(1 - AR)**2/x_init.shape[0], 1/6),
                                                        lambda _: step_sizes * max_freq, 
                                                        operand = None)
        if not compute_freqs or S > 2:
            S_freq = jnp.max(jnp.array([1, 2/step_size * jnp.power((2*jnp.pi*(1 - AR)**2)/jnp.sum(frequencies**6), 1/6)]))
            std_dev_freq = jnp.std(frequencies)
            if std_dev_freq <= 1:
                fitting_factor = S_freq
                t_ColSI = stage/(max_freq * fitting_factor)
                t_lower = h_lower/(max_freq * fitting_factor)
                # stability_limit = 2*stage/(max_freq * fitting_factor)
                step_sizes = jax.random.uniform(key, shape = (n_samples_prod, )) * (t_ColSI - t_lower) + t_lower
                dimensionless_step_sizes = jax.lax.cond(S_freq > 1, 
                                                        lambda _: (2*max_freq*step_sizes/step_size)*jnp.power(2*jnp.pi*(1 - AR)**2/(jnp.sum(frequencies**6)), 1/6),
                                                        lambda _: step_sizes * max_freq, 
                                                        operand = None)
            elif std_dev_freq > 1:
                fitting_factor = S_freq
                smooth_max_freq = jnp.maximum(max_freq - std_dev_freq, 1e-8)
                t_ColSI = stage/(S_freq * smooth_max_freq)
                t_lower = h_lower/(S_freq * smooth_max_freq)
                # stability_limit = 2*stage/(S_freq * (max_freq - std_dev_freq))
                step_sizes = jax.random.uniform(key, shape = (n_samples_prod, )) * (t_ColSI - t_lower) + t_lower
                dimensionless_step_sizes = jax.lax.cond(S_freq > 1, 
                                                        lambda _: (2*smooth_max_freq*step_sizes/step_size)*jnp.power(2*jnp.pi*(1 - AR)**2/(jnp.sum(frequencies**6)), 1/6),
                                                        lambda _: step_sizes * smooth_max_freq, 
                                                        operand = None)
    return dimensionless_step_sizes, step_sizes, fitting_factor, x_final, p_final, key

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

def _sAIA_OptimalCoeffs(dimensionless_step_sizes, stage, key, n_coeff_samples = 256, n_h_samples = 256):
    """
    Compute optimal coefficients for s-AIA method
    -------------------------
    Parameters:
    dimensionless_step_sizes (jax.Array): dimensionless step sizes
    stage (int): number of stages
    key (int): random number generator key
    n_coeff_samples (int): number of coefficient samples
    """
    if stage == 2:
        rho = _rho_2
        b_low, b_high = ME_2().b, VV_2().b
    elif stage == 3:
        rho = _rho_3
        b_low, b_high = ME_3().b, VV_3().b
    else:
        raise NotImplementedError("Only 2- & 3-stage integrators are supported as of now.")

    del key
    dimensionless_step_sizes = np.asarray(dimensionless_step_sizes, dtype = float)
    optimal_coeffs = []

    b_values = jnp.linspace(b_low, b_high, n_coeff_samples)
    for h_max in dimensionless_step_sizes:
        h_max = float(np.clip(h_max, 1e-6, 2 * stage - 1e-6))
        h_values = jnp.linspace(1e-6, h_max, n_h_samples)
        rho_grid = jax.vmap(lambda b: jax.vmap(lambda h: rho(h, b))(h_values))(b_values)
        rho_grid = jnp.where(jnp.isfinite(rho_grid) & (rho_grid >= 0), rho_grid, jnp.inf)
        worst_case = jnp.max(rho_grid, axis = 1)
        optimal_coeffs.append(float(b_values[jnp.argmin(worst_case)]))

    return jnp.asarray(optimal_coeffs)

def __lambda_phi(stage, a = None, b = None):
    if stage == 2:
        lambda_2 = (6*b - 1)/24
        return lambda_2
    elif stage == 3:
        lambda_3 = (1 - 6*a * (1 - a) * (1 - 2*b))/12
        return lambda_3

def optimal_momentum_noise(step_size_nondim, stage, D, a = None, b = None):
    """
    Compute optimal momentum noise for GHMC sampler
    -------------------------
    Parameters:
    step_size_nondim (float): dimensionless step size
    stage (int): number of stages
    D (int): dimension of data
    a (float): coefficient for 3-stage integrator
    b (float): coefficient for 3-stage integrator
    -------------------------
    Returns:
    phi_opt (float): optimal momentum noise
    """
    lambda_phi_val = __lambda_phi(stage, a, b)
    phi_opt = jnp.minimum(1, -jnp.log(0.999)/D * (1 + 2 * step_size_nondim ** 2 * lambda_phi_val)/(2 * step_size_nondim**4 * lambda_phi_val ** 2))
    return phi_opt

def sAIA(x_init, potential_args, n_samples_tune, n_samples_check, 
         n_samples_burn_in, n_samples_prod, potential, mass_matrix, 
         target_AR = 0.92, stage = 2, sensibility = 0.01, 
         delta_step = 0.01, compute_freqs = True, compute_hessian = True, sampler = "HMC", RNG_key = 42):
    """
    s-AIA: Adaptive Integration Approach for Computation Statistics.

    Note: As of this version the s-AIA method is only supported for 2- & 3-stage
    Splitting Integrators w/ HMC, GHMC sampling.
    -------------------------
    Parameters:
        x_init (jax.Array): initial position
        potential_args (tuple): arguments for Hamiltonian potential
        n_samples_tune (int): number of samples for tuning
        n_samples_check (int): number of samples for checking acceptance rate
        n_samples_burn_in (int): number of samples for burn-in
        n_samples_prod (int): number of samples for production
        potential (function): Hamiltonian potential
        mass_matrix (jax.Array): mass matrix
        target_AR (float): target acceptance rate
        stage (int): number of stages
        sensibility (float): sensibility for acceptance rate
        delta_step (float): step size increment/decrement
        compute_freqs (bool): compute frequencies
        compute_hessian (bool): compute hessian function
        sampler (str): sampler type
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
    print(f"{'Compute Hessian':^30}|{('Yes' if compute_hessian else 'No'):^30}")
    print("="*61)
    potential = jax.jit(jax.tree_util.Partial(potential, *potential_args))
    potential_grad = jax.jit(jax.grad(potential))
    if compute_hessian == True: potential_hessian = jax.jit(jax.hessian(potential))
    else: potential_hessian = None 
    key = jax.random.PRNGKey(RNG_key)
    # Step 1: Tuning Stage
    print("1) Tuning Stage...")
    n_samples, step_size, n_steps, integrator = n_samples_tune, 1/x_init.shape[0], 1, VerletIntegrator()
    momentum_noise_lower, momentum_noise_upper = None, None
    if sampler == "GHMC":
        if stage == 2: # TODO: FIX momentum assignation for 2-stage
            a, b = 0, 1/4
            momentum_noise_lower = optimal_momentum_noise(stage, stage, x_init.shape[0], a, b)
            momentum_noise_upper = optimal_momentum_noise(1.5, stage, x_init.shape[0], a, b)
        elif stage == 3:
            b_low_phi = 0.11888010966548;
            a_low_phi = (1-2*b_low_phi)/(4*(1-3*b_low_phi));
            b_up_phi = 0.113252;
            a_up_phi = (1-2*b_up_phi)/(4*(1-3*b_up_phi));
            
            momentum_noise_lower = optimal_momentum_noise(stage, stage, x_init.shape[0], a_low_phi, b_low_phi)
            momentum_noise_upper = optimal_momentum_noise(2.0772, stage, x_init.shape[0], a_up_phi, b_up_phi)
    print(f"\t- Number of Tuning Samples: {n_samples}")
    print(f"\t- Dimension of Data: {x_init.shape[0]}")
    print(f"\t- Initial Step-Size: {step_size}")
    if sampler == "GHMC":
        print(f"\t- Initial Momentum Noise (Lower Bound): {momentum_noise_lower}")
        print(f"\t- Initial Momentum Noise (Upper Bound): {momentum_noise_upper}")
    tuned_step_size, AR_tuned, x_tuned, p_tuned, key = _sAIA_Tuning(
        x_init, n_samples, n_samples_check, step_size, n_steps, sensibility, target_AR,
        potential, potential_grad, potential_hessian, mass_matrix, delta_step, integrator,
        sampler, momentum_noise_lower, momentum_noise_upper, key
    )
    print(f"\t- Tuned AR: {AR_tuned}")
    print(f"\t- Tuned Step-Size: {tuned_step_size:.5f}")
    print("="*61)
    # Step 2: Burn-In Stage
    print("2) Burn-In Stage...")
    n_samples, step_size = n_samples_burn_in, tuned_step_size
    print(f"\t- Number of Burn-In Samples: {n_samples}")
    dimensionless_step_sizes, step_sizes, fitting_factor, x_burned, p_burned, key = _sAIA_BurnIn(
        x_tuned, n_samples, n_samples_prod, compute_freqs, step_size, n_steps, stage, potential,
        potential_grad, potential_hessian, mass_matrix, integrator, sampler, momentum_noise_lower,
        momentum_noise_upper, key, p_init = p_tuned
    )
    print(f"\t- Fitting factor: {fitting_factor}")
    print(f"\t- Dimensionless Step-Sizes: {dimensionless_step_sizes}")
    print(f"\t- Step-Sizes: {step_sizes}")
    opt_integration_coeffs = _sAIA_OptimalCoeffs(dimensionless_step_sizes, stage, RNG_key)
    print(f"\t- Optimal Integration Coefficients: {opt_integration_coeffs}")
    print("="*61)
    # Step 3: Production Stage
    print("3) Production Stage...")
    n_steps = jnp.clip(jnp.rint(1.0 / step_sizes), a_min = 1, a_max = 100).astype(int)
    print(f"\t- Number of Steps: {n_steps}")
    if stage == 2:
        integrator = [(2, float(b)) for b in opt_integration_coeffs]
    elif stage == 3:
        a_coeffs = [(2*b - 1)/(2*(6*b-2)) for b in opt_integration_coeffs]
        integrator = [(3, float(a), float(b)) for a, b in zip(a_coeffs, opt_integration_coeffs)]
    assert len(integrator) == n_samples_prod, "Number of integrators must be equal to number of samples"
    if sampler == "HMC":
        samples, N_acc_prod, _, _, _, key = _sAIA_HMC(
            x_burned, n_samples = n_samples_prod, burn_in = 0, step_size = step_sizes,
            n_steps = n_steps, potential = potential, potential_grad = potential_grad,
            potential_hessian = potential_hessian, mass_matrix = mass_matrix,
            integrator = integrator, key = key, phase_name = "Production", return_state = True
        )
    elif sampler == "GHMC":
        samples, N_acc_prod, _, _, _, key = _sAIA_GHMC_stateful(
            x_burned, p_burned, n_samples = n_samples_prod, burn_in = 0, step_size = step_sizes,
            n_steps = n_steps, potential = potential, potential_grad = potential_grad,
            potential_hessian = potential_hessian, mass_matrix = mass_matrix,
            momentum_noise_lower = momentum_noise_lower, momentum_noise_upper = momentum_noise_upper,
            integrator = integrator, key = key, phase_name = "Production", return_state = True
        )
    AR_prod = acceptance_rate(N_acc_prod, n_samples_prod)
    print("="*61)
    print(f"Production stage finished, production acceptance rate: {AR_prod}.")
    return samples

def _sMAIA_MHMC(x_init, n_samples, burn_in, step_size, n_steps, 
                potential, potential_grad, potential_hessian, mass_matrix, 
                momentum_noise, integrator, order, key, phase_name):
    """
    Single-Chain Modified Hamiltonian Monte-Carlo (MHMC) sampler for s-MAIA.
    This function is adapted for the adaptive framework, handling dynamic
    step sizes, integrators, and momentum noise.
    """
    # Ensure step_size, n_steps, and integrator are lists for adaptive sampling
    if isinstance(step_size, (int, float)):
        step_size = [step_size] * n_samples
    if isinstance(n_steps, int):
        n_steps = [n_steps] * n_samples
    if isinstance(integrator, Integrator):
        integrator = [integrator] * n_samples
    if isinstance(momentum_noise, (int, float)):
        momentum_noise = [momentum_noise] * n_samples

    samples = []
    weights = []
    frequencies = []
    acceptances = 0
    x = x_init
    # Initial momentum (gaussian), shape given by mass matrix
    p = jax.random.multivariate_normal(key, jnp.zeros(x.shape[0]), mass_matrix)
    for n in tqdm(range(n_samples + burn_in), desc=f"\t- Running {phase_name} Phase MHMC", ncols=100):
        key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
        # Get current adaptive parameters for this step
        current_idx = min(n - burn_in, n_samples - 1) if n >= burn_in else 0
        current_step_size = step_size[current_idx]
        current_n_steps = n_steps[current_idx]
        current_integrator = integrator[current_idx]
        current_momentum_noise = momentum_noise[current_idx]
        # Get coefficients for the Modified Hamiltonian
        c = get_mhmc_coeffs(order=order, stage=current_integrator.stage, 
                            b=current_integrator.b, a=getattr(current_integrator, 'a', None))
        # Sample noise vector
        mu = jax.random.multivariate_normal(subkey1, jnp.zeros(x.shape[0]), mass_matrix)
        # Propose updated momentum and noise vector
        p_prop = jnp.sqrt(1 - current_momentum_noise) * p + jnp.sqrt(current_momentum_noise) * mu
        mu_prop = -jnp.sqrt(current_momentum_noise) * p + jnp.sqrt(1 - current_momentum_noise) * mu
        # Compute Modified Hamiltonians
        H_old = Modified_Hamiltonian(x, p, potential, potential_grad, potential_hessian, mass_matrix, current_step_size, order, c)
        H_prop = Modified_Hamiltonian(x, p_prop, potential, potential_grad, potential_hessian, mass_matrix, current_step_size, order, c)
        # Compute Difference of Extended Hamiltonians
        dH_ext = Extended_Hamiltonian(H_prop, mu_prop, mass_matrix) - Extended_Hamiltonian(H_old, mu, mass_matrix)
        # Metropolis-Hastings acceptance (for momentum)
        accept_momentum = jax.random.uniform(subkey2) < jnp.exp(-dH_ext)
        p_accepted = jax.lax.cond(accept_momentum, lambda: p_prop, lambda: p)
        # Integrate Hamiltonian dynamics with accepted momentum
        x_new, p_new = current_integrator.integrate(x, p_accepted, potential_grad, current_n_steps, mass_matrix, current_step_size)
        # Compute Difference of Modified Hamiltonians
        H_initial_step = Modified_Hamiltonian(x, p_accepted, potential, potential_grad, potential_hessian, mass_matrix, current_step_size, order, c)
        H_final_step = Modified_Hamiltonian(x_new, p_new, potential, potential_grad, potential_hessian, mass_matrix, current_step_size, order, c)
        dH = H_final_step - H_initial_step
        # Metropolis-Hastings acceptance (for position)
        accept_position = jax.random.uniform(subkey3) < jnp.exp(-dH)
        x, p = jax.lax.cond(accept_position, lambda: (x_new, p_new), lambda: (x, -p_accepted))
        if n >= burn_in:
            acceptances += jax.lax.cond(accept_position, lambda: 1, lambda: 0)
            samples.append(x)
            # Compute importance sampling weights
            weight = jnp.exp(Hamiltonian(x, p, potential, mass_matrix) - 
                             Modified_Hamiltonian(x, p, potential, potential_grad, potential_hessian, mass_matrix, current_step_size, order, c))
            weights.append(weight)
            # Compute frequencies for adaptive tuning
            Hessian = potential_hessian(x)
            frequencies.append(_compute_frequencies(Hessian))
    return (jnp.stack(samples), jnp.stack(weights), acceptances, jnp.stack(frequencies) if frequencies else jnp.array([]))

def sMAIA(x_init, potential_args, n_samples_tune, n_samples_check, 
          n_samples_burn_in, n_samples_prod, potential, mass_matrix, 
          target_AR=0.92, stage=2, sensibility=0.01, 
          delta_step=0.01, compute_freqs=True, order=4, RNG_key=42):
    """
    s-MAIA: Standalone implementation of s-AIA for the MHMC sampler.

    This function follows the three-phase adaptive integration scheme:
    1. Tuning: Finds an optimal initial step-size.
    2. Burn-In: Estimates system frequencies to create adaptive parameters.
    3. Production: Runs the final MHMC sampling with adaptive step-sizes
       and integrators.
   
    Note: As of this version the s-MAIA method is only supported for 2- & 3-stage
    Splitting Integrators with Modified Hamiltonian Monte-Carlo (MHMC) sampling.
    -------------------------
    Parameters:
        x_init (jax.Array): initial position
        potential_args (tuple): arguments for Hamiltonian potential
        n_samples_tune (int): number of samples for tuning
        n_samples_check (int): number of samples for checking acceptance rate
        n_samples_burn_in (int): number of samples for burn-in
        n_samples_prod (int): number of samples for production
        potential (function): Hamiltonian potential
        mass_matrix (jax.Array): mass matrix
        target_AR (float): target acceptance rate
        stage (int): number of stages (2 or 3)
        sensibility (float): sensibility for acceptance rate
        delta_step (float): step size increment/decrement
        compute_freqs (bool): compute frequencies for adaptive tuning
        order (int): order of the Modified Hamiltonian (default is 4)
        RNG_key (int): random number generator key
    -------------------------
    Returns:
        samples (jax.Array): samples from the MHMC sampler
    """
    print("Running s-MAIA (s-AIA for MHMC) Adaptive Integration Scheme...")
    print("="*61)
    print(f"{'Sampler':^30}|{'MHMC':^30}")
    print(f"{'Num. Samples Tune':^30}|{n_samples_tune:^30}")
    print(f"{'Num. Samples Check':^30}|{n_samples_check:^30}")
    print(f"{'Num. Samples Burn-In':^30}|{n_samples_burn_in:^30}")
    print(f"{'Num. Samples Prod':^30}|{n_samples_prod:^30}")
    print(f"{'Stage':^30}|{stage:^30}")
    print(f"{'Target AR':^30}|{target_AR:^30}")
    print(f"{'Sensibility':^30}|{sensibility:^30}")
    print(f"{'Delta Step':^30}|{delta_step:^30}")
    print(f"{'Compute Freqs':^30}|{('Yes' if compute_freqs else 'No'):^30}")
    print(f"{'Order of Modified Hamiltonian':^30}|{order:^30}")
    print("="*61)

    # JIT compile potential and its derivatives
    potential = jax.jit(jax.tree_util.Partial(potential, *potential_args))
    potential_grad = jax.jit(jax.grad(potential))
    potential_hessian = jax.jit(jax.hessian(potential))
    key = jax.random.PRNGKey(RNG_key)

    # === 1. Tuning Stage ===
    print("1) Tuning Stage...")
    # Use GHMC for efficient initial tuning
    initial_step_size, n_steps = 1.0 / x_init.shape[0], 1
    if stage == 2: a_tune, b_tune = 0, 1/4
    elif stage == 3: a_tune, b_tune = 1/3, 1/6
    print(f"\t- Number of Tuning Samples: {n_samples_tune}")
    print(f"\t- Dimension of Data: {x_init.shape[0]}")
    print(f"\t- Initial Step-Size: {initial_step_size}")
    
    momentum_noise_lower = optimal_momentum_noise(stage, stage, x_init.shape[0], a_tune, b_tune)
    momentum_noise_upper = optimal_momentum_noise(1.5 if stage == 2 else 2.0772, stage, x_init.shape[0], a_tune, b_tune)
    
    print(f"\t- Initial Momentum Noise (Lower Bound): {momentum_noise_lower}")
    print(f"\t- Initial Momentum Noise (Upper Bound): {momentum_noise_upper}")
    # Tune the step size using the adaptive tuning function
    tuned_step_size, AR_tuned, x_tuned, p_tuned, key = _sAIA_Tuning(
        x_init, n_samples_tune, n_samples_check, initial_step_size, n_steps, sensibility,
        target_AR, potential, potential_grad, potential_hessian, mass_matrix,
        delta_step, VerletIntegrator(), "GHMC", momentum_noise_lower,
        momentum_noise_upper, key)
    print(f"\t- Tuned AR: {AR_tuned}")
    print(f"\t- Tuned Step-Size: {tuned_step_size}")
    print("="*61)
    # === 2. Burn-In Stage ===
    print("2) Burn-In Stage...")
    # Run MHMC to estimate frequencies with the tuned step size
    burn_in_integrator = VerletIntegrator()
    # Use frequency information to compute adaptive parameters for the production run
    dimensionless_step_sizes, step_sizes, _, x_burned, p_burned, key = _sAIA_BurnIn(
         x_tuned, n_samples_burn_in, n_samples_prod, compute_freqs, tuned_step_size, n_steps,
         stage, potential, potential_grad, potential_hessian, mass_matrix,
         burn_in_integrator, "GHMC", momentum_noise_lower, momentum_noise_upper, key, p_init = p_tuned)
    
    opt_integration_coeffs = _sAIA_OptimalCoeffs(dimensionless_step_sizes, stage, RNG_key)
    print("\t- Dimensionless Step-Sizes computed.")
    print("\t- Optimal Integration Coefficients computed.")
    print("="*61)

    # === 3. Production Stage ===
    print("3) Production Stage...")
    key, subkey1, subkey2 = jax.random.split(key, 3)
    
    # Create adaptive integrators and step numbers for the production run
    prod_steps = jax.random.randint(subkey1, shape=(n_samples_prod,), minval=1, maxval=jnp.maximum(2, 2 * (x_init.shape[0] / step_sizes) - 1))
    
    if stage == 2:
        prod_integrators = [MSSI_2(b) for b in opt_integration_coeffs]
    else: # stage == 3
        a_coeffs = [(2 * b - 1) / (2 * (6 * b - 2)) for b in opt_integration_coeffs]
        prod_integrators = [MSSI_3(a, b) for a, b in zip(a_coeffs, opt_integration_coeffs)]
    
    # Use adaptive momentum noise for production run
    momentum_noise_prod = jax.random.uniform(subkey2, shape=(n_samples_prod,)) * (momentum_noise_upper - momentum_noise_lower) + momentum_noise_lower

    # Run the final production sampling with all adaptive parameters
    samples, weights, _, _ = _sMAIA_MHMC(
        x_burned, n_samples_prod, 100, step_sizes, prod_steps,
        potential, potential_grad, potential_hessian, mass_matrix,
        momentum_noise_prod, prod_integrators, order, key, "Production")
    
    print("="*61)
    print("s-MAIA sampling complete.")
    
    return samples, weights
