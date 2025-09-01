import jax
import jax.numpy as jnp
from tqdm import tqdm
from functools import partial
from ..utils.hamiltonian import Hamiltonian
from ..integrators.integrators import VerletIntegrator, VV_2, ME_2, VV_3, ME_3, MSSI_2, MSSI_3, Integrator
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
    potential_hessian (function): Hamiltonian potential Hessian
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
            if potential_hessian is not None:
                Hessian = potential_hessian(x)
                freqs_iter = _compute_frequencies(Hessian)
                frequencies.append(freqs_iter)
            else: frequencies = jax.numpy.ones(x.shape[0])
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
    potential_hessian (function): Hamiltonian potential Hessian
    mass_matrix (jax.Array): mass matrix
    momentum_noise_lower (float): lower bound for momentum noise
    momentum_noise_upper (float): upper bound for momentum noise
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
            if potential_hessian is not None:
                Hessian = potential_hessian(x)
                freqs_iter = _compute_frequencies(Hessian)
                frequencies.append(freqs_iter)
            else: frequencies = jax.numpy.ones(x.shape[0])
    samples, frequencies = jnp.stack(samples, axis=0), jnp.stack(frequencies, axis=0)
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
    return tuned_step_size, AR

def _sAIA_BurnIn(x_init, n_samples_burn_in, n_samples_prod, compute_freqs, step_size, 
                 n_steps, stage, potential, potential_grad, potential_hessian, 
                 mass_matrix, integrator, sampler, momentum_noise_lower,
                 momentum_noise_upper, key):
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
                t_ColSI = stage/(S_freq * (max_freq - std_dev_freq))
                t_lower = h_lower/(S_freq * (max_freq - std_dev_freq))
                # stability_limit = 2*stage/(S_freq * (max_freq - std_dev_freq))
                step_sizes = jax.random.uniform(key, shape = (n_samples_prod, )) * (t_ColSI - t_lower) + t_lower
                dimensionless_step_sizes = jax.lax.cond(S_freq > 1, 
                                                        lambda _: (2*(max_freq - std_dev_freq)*step_sizes/step_size)*jnp.power(2*jnp.pi*(1 - AR)**2/(jnp.sum(frequencies**6)), 1/6),
                                                        lambda _: step_sizes * (max_freq - std_dev_freq), 
                                                        operand = None)
    return dimensionless_step_sizes, step_sizes, fitting_factor

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
    """
    Compute optimal coefficients for s-AIA method
    -------------------------
    Parameters:
    dimensionless_step_sizes (jax.Array): dimensionless step sizes
    stage (int): number of stages
    key (int): random number generator key
    n_coeff_samples (int): number of coefficient samples
    """
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
    # Step 1: Tuning Stage
    print("1) Tuning Stage...")
    n_samples, step_size, n_steps, integrator = n_samples_tune, 1/x_init.shape[0], 1, VerletIntegrator()
    momentum_noise_lower, momentum_noise_upper = None, None
    if sampler == "GHMC":
        if stage == 2: # TODO: FIX momentum assignation for 2-stage
            a, b = 0, 1/4
            momentum_noise_lower = optimal_momentum_noise(stage, stage, x_init.shape[0], a, b)
            momentum_noise_upper = optimal_momentum_noise(2.0772, stage, x_init.shape[0], a, b)
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
    tuned_step_size, AR_tuned = _sAIA_Tuning(x_init, n_samples, n_samples_check, step_size, n_steps, 
                                   sensibility, target_AR, potential, potential_grad, potential_hessian,
                                   mass_matrix, delta_step, integrator, sampler, momentum_noise_lower, momentum_noise_upper, jax.random.PRNGKey(RNG_key))
    print(f"\t- Tuned AR: {AR_tuned}")
    print(f"\t- Tuned Step-Size: {tuned_step_size:.5f}")
    print("="*61)
    # Step 2: Burn-In Stage
    print("2) Burn-In Stage...")
    n_samples, step_size = n_samples_burn_in, tuned_step_size
    print(f"\t- Number of Burn-In Samples: {n_samples}")
    dimensionless_step_sizes, step_sizes, fitting_factor = _sAIA_BurnIn(x_init, n_samples, n_samples_prod, compute_freqs, step_size, n_steps, 
                                                        stage, potential, potential_grad, potential_hessian, mass_matrix, 
                                                        integrator, sampler, momentum_noise_lower, 
                                                        momentum_noise_upper, jax.random.PRNGKey(RNG_key))
    print(f"\t- Fitting factor: {fitting_factor}")
    print(f"\t- Dimensionless Step-Sizes: {dimensionless_step_sizes}")
    print(f"\t- Step-Sizes: {step_sizes}")
    opt_integration_coeffs = _sAIA_OptimalCoeffs(dimensionless_step_sizes, stage, RNG_key)
    print(f"\t- Optimal Integration Coefficients: {opt_integration_coeffs}")
    print("="*61)
    # Step 3: Production Stage
    print("3) Production Stage...")
    # n_steps = jax.random.randint(jax.random.PRNGKey(RNG_key), shape=(n_samples_prod,), minval=1, maxval=2 * (x_init.shape[0] / step_sizes) - 1)
    if fitting_factor <= 1.3 and compute_hessian == True: n_steps = jax.random.randint(jax.random.PRNGKey(RNG_key), shape=(n_samples_prod,), minval=1, maxval=1)
    elif fitting_factor > 1.3 and compute_hessian == True: n_steps = jax.random.randint(jax.random.PRNGKey(RNG_key), shape=(n_samples_prod,), minval=2, maxval=6)
    elif compute_hessian == False: n_steps = jax.random.randint(jax.random.PRNGKey(RNG_key), shape=(n_samples_prod,), minval=1, maxval=1)
    print(f"\t- Number of Steps: {n_steps}")
    if stage == 2:
        integrator = [MSSI_2(b) for b in opt_integration_coeffs]
    elif stage == 3:
        a_coeffs = [(2*b - 1)/(2*(6*b-2)) for b in opt_integration_coeffs]
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