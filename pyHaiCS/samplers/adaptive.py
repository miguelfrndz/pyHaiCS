import warnings
import jax
import numpy as np
import jax.numpy as jnp
from tqdm import tqdm
from functools import partial
from .hamiltonian import _run_chain, make_hmc_step, make_ghmc_step, make_mhmc_step
from ..utils.hamiltonian import Position_Terms, as_mass
from ..integrators.integrators import VerletIntegrator, VV_2, ME_2, VV_3, ME_3, MSSI_2, MSSI_3, Integrator, get_mhmc_coeffs, _rho_2, _rho_3, _optimal_b, sAIA_2, sAIA_3
from ..utils.metrics import acceptance_rate

@jax.jit
def _compute_frequencies(Hessian, mass_matrix = None):
    """
    Compute frequencies of a Hamiltonian system in ascending order.
    -------------------------
    Parameters:
        Hessian (jax.Array): Hessian matrix
        mass_matrix (jax.Array): mass matrix
    -------------------------
    Returns:
        freqs (jax.Array): frequencies
    """
    return jnp.sqrt(jnp.abs(jnp.linalg.eigvalsh(Hessian)))

def _pad_front(tree, n):
    """
    Repeat n times the first element of every leaf, along the leading axis.
    Example:
        For n = 2 and an array leaf:
        [1, 2, 3] -> [1, 1, 1, 2, 3]
    -------------------------
    Parameters:
        tree (pytree): arrays with the same leading length
        n (int): number of repetitions
    -------------------------
    Returns:
        tree (pytree): same structure, with the leading length increased by n
    """
    # a[:1] preserves the original rank (shape (1, ...)) unlike a[0].
    # Repeat this value n times along the leading axis (axis 0).
    return jax.tree_util.tree_map(
        lambda a: jnp.concatenate([jnp.repeat(a[:1], n, axis=0), a]), tree)

def _per_iter(sampler, n, step_size, n_steps, integrator, key, phi_lo=None, phi_hi=None):
    """
    Build the per iteration inputs (xs) for _run_chain.
    -------------------------
    Parameters:
        sampler (str): "HMC" or "GHMC"
        n (int): number of iterations
        step_size (float or jax.Array): step size, a scalar or an array of length n
        n_steps (int or jax.Array): number of integration steps, a scalar or an array of length n
        integrator (object): integrator pytree
        key (jax.Array): PRNG key, used to draw the momentum noise (GHMC only)
        phi_lo (float): lower bound of the momentum noise (GHMC only)
        phi_hi (float): upper bound of the momentum noise (GHMC only)
    -------------------------
    Returns:
        per_iter (tuple): (hs, Ls, integrator) for HMC, (hs, Ls, phis, integrator) for GHMC
    """
    hs = jnp.broadcast_to(jnp.asarray(step_size), (n,))
    Ls = jnp.broadcast_to(jnp.asarray(n_steps), (n,))
    if sampler == "GHMC":
        phis = jax.random.uniform(key, (n,)) * (phi_hi - phi_lo) + phi_lo
        return (hs, Ls, phis, integrator)
    return (hs, Ls, integrator)

def _sAIA_run(x_init, n_samples, burn_in, per_iter, step_fn, sampler,
              mass_matrix, potential_hessian, key, freq_thin=1, progress=True, state0=None):
    """
    Run n_samples + burn_in iterations with lax.scan and collect samples, acceptances and frequencies.
    -------------------------
    Parameters:
        x_init (jax.Array): initial position, also used to get the dimension
        n_samples (int): number of samples
        burn_in (int): burn-in iterations, run with the parameters of the first iteration and discarded
        per_iter (tuple): per iteration inputs, as built by _per_iter 
        step_fn (function): one chain iteration, built by make_hmc_step or make_ghmc_step
        sampler (str): "HMC" or "GHMC"
        mass_matrix (jax.Array): mass matrix
        potential_hessian (function or None): Hessian of the potential, None skips the frequencies
        key (jax.Array): PRNG key
        freq_thin (int): compute the frequencies on one sample every freq_thin
        progress (bool): show a tqdm progress bar
        state0 (pytree or None): initial state of the chain, the position x for HMC or the pair (x, p) for GHMC. If None, the chain starts from x_init
    -------------------------
    Returns:
        samples (jax.Array): samples after the burn-in
        N_acc (int): number of accepted proposals after the burn-in
        frequencies (jax.Array): frequencies computed from the Hessians (a row of ones if potential_hessian is None)
        state (pytree): final state of the chain
    """
    if state0 is None:
        if sampler == "GHMC":
            key, k0 = jax.random.split(key)
            p0 = as_mass(mass_matrix).sample(k0)
            state0 = (x_init, p0)
        else:
            state0 = x_init
    state_f, (traj, accepted) = _run_chain(state0, key, n_samples + burn_in, step_fn,
                                _pad_front(per_iter, burn_in), progress=progress, return_state=True)
    samples, accepted = traj[burn_in:], accepted[burn_in:]
    if potential_hessian is None:
        frequencies = jnp.ones((1, x_init.shape[0]))     
    else:
        frequencies = jax.lax.map(
            lambda x: _compute_frequencies(potential_hessian(x)), samples[::freq_thin])
    return samples, accepted.sum(), frequencies, state_f

def _sAIA_Tuning(x_init, n_samples_tune, n_samples_check, step_size, n_steps, sensibility,
                 target_AR, mass_matrix, step_factor, integrator, sampler,
                 momentum_noise_lower, momentum_noise_upper, key, step_fn, progress=True):
    """
    Tuning stage for the s-AIA method
    -------------------------
    Parameters:
    n_samples_tune (int): number of samples for tuning
    n_samples_check (int): number of samples for checking acceptance rate
    step_size (float): initial step size
    n_steps (int): number of integration steps
    sensibility (float): half width of the target AR interval
    target_AR (float): target acceptance rate
    mass_matrix (jax.Array): mass matrix
    step_factor (float): step size relative increment/decrement 
    integrator (object): integrator object
    sampler (str): sampler type
    momentum_noise_lower (float): lower bound for momentum noise
    momentum_noise_upper (float): upper bound for momentum noise
    key (int): random number generator key
    step_fn (function): one chain iteration, built by make_hmc_step or make_ghmc_step
    progress (bool): show the progress bar
    -------------------------
    Returns:
    tuned_step_size (float): tuned step size
    AR (float): last measured acceptance rate
    """
    tuned_step_size, N_tot, last_direction = step_size, 0, 0
    pbar = tqdm(total=n_samples_tune, desc="Tuning", unit="it")
    state=None
    while N_tot + n_samples_check <= n_samples_tune:
        key, k_noise, k_run = jax.random.split(key, 3)
        per_iter = _per_iter(sampler, n_samples_check, tuned_step_size, n_steps, integrator,
                             k_noise, momentum_noise_lower, momentum_noise_upper)
        _, N_acc, _, state = _sAIA_run(x_init, n_samples_check, 0, per_iter, step_fn, sampler,
                                mass_matrix, None, k_run, progress=False, state0=state)
        AR = acceptance_rate(N_acc, n_samples_check)
        N_tot += n_samples_check
        pbar.update(n_samples_check)
        pbar.set_postfix(AR=f"{AR:.3f}", h=f"{tuned_step_size:.5f}")
        if abs(AR - target_AR) <= sensibility:
            break
        direction = 1 if AR < target_AR else -1      
        if direction == -last_direction:
            step_factor /= 2     # if at some point we change direction, we half the step factor
        last_direction = direction
        tuned_step_size *= 1.0 - direction * step_factor
    pbar.close()
    return tuned_step_size, AR, N_tot, state

# Lower end h_lower of the dimensionless step size interval, for each supported stage
# (2 stages: arbitrary value, should be changed)
_H_LOWER = {2: 1.5, 3: 2.0772}

def _draw_step_sizes(key, n, stage, scale):
    """
    Draw n step sizes uniformly in [h_lower / scale, stage / scale] (t_lower, t_ColSI).
    -------------------------
    Parameters:
        key (jax.Array): PRNG key
        n (int): number of step sizes
        stage (int): number of stages (2 or 3)
        scale (float): scaling of the interval, i.e. fitting factor * frequency scale
    -------------------------
    Returns:
        step_sizes (jax.Array): array of shape (n,)
    """
    t_lower, t_ColSI = _H_LOWER[stage] / scale, stage / scale
    return jax.random.uniform(key, shape=(n,)) * (t_ColSI - t_lower) + t_lower

def _sAIA_BurnIn(x_init, n_samples_burn_in, n_samples_prod, compute_freqs, step_size,
                 n_steps, stage, potential_hessian, mass_matrix, integrator, sampler,
                 momentum_noise_lower, momentum_noise_upper, key, step_fn, progress=True, state0=None):
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
    potential_hessian (function): Hamiltonian potential Hessian
    mass_matrix (jax.Array): mass matrix
    integrator (object): integrator object
    sampler (str): sampler type
    momentum_noise_lower (float): lower bound for momentum noise
    momentum_noise_upper (float): upper bound for momentum noise
    key (int): random number generator key
    step_fn (function): one chain iteration, built by make_hmc_step or make_ghmc_step
    progress (bool): show the progress bar
    state0 (pytree or None): initial state of the chain
    -------------------------
    Returns:
    dimensionless_step_sizes (jax.Array): dimensionless step sizes
    step_sizes (jax.Array): production step sizes, drawn at random in the interval [t_lower, t_ColSI]
    fitting_factor (float): scaling factor of the step size interval
    c (float): constant estimated in the burn in, so that the dimensionless step size is c * step_sizes
    """
    if stage not in _H_LOWER:
        raise NotImplementedError("Only 2- & 3-stage integrators are supported as of now.")
    per_iter = _per_iter(sampler, n_samples_burn_in, step_size, n_steps, integrator,
                        key, momentum_noise_lower, momentum_noise_upper)
    _, N_acc, frequencies, state = _sAIA_run(x_init, n_samples_burn_in, 0, per_iter, step_fn, sampler,
                                            mass_matrix, potential_hessian, key,
                                            freq_thin=max(1, n_samples_burn_in // 100), progress=progress, state0=state0)
    # Average over the samples (frequencies are real and sorted)
    frequencies = jnp.mean(frequencies, axis=0)
    max_freq = jnp.max(frequencies)
    AR = acceptance_rate(N_acc, n_samples_burn_in)
    g = jnp.power(2*jnp.pi*(1 - AR)**2 / x_init.shape[0], 1/6)

    # Each case below only sets three things:
    #   fitting_factor: scaling factor of the step size interval
    #   scale: the step size interval is [h_lower/scale, stage/scale]
    #   c: constant such that dimensionless step size = c * step size
    if potential_hessian is None:
        # max_freq = 1
        S = jnp.maximum(1, 2/step_size * g)
        fitting_factor, scale = S, S
        c = jnp.where(S > 1, 2/step_size * g, max_freq)
    else:
        S = jnp.maximum(1, 2/(max_freq * step_size) * g) if compute_freqs else None
        if S is not None and S <= 2:
            fitting_factor, scale = S, max_freq * S
            c = jnp.where(S > 1, 2/step_size * g, max_freq)
        else:
            g_freq = jnp.power(2*jnp.pi*(1 - AR)**2 / jnp.sum(frequencies**6), 1/6)
            S_freq = jnp.maximum(1, 2/step_size * g_freq)
            std_dev_freq = jnp.std(frequencies)
            f_eff = max_freq if std_dev_freq <= 1 else max_freq - std_dev_freq
            fitting_factor, scale = S_freq, f_eff * S_freq
            c = jnp.where(S_freq > 1, (2*f_eff/step_size) * g_freq, f_eff)

    step_sizes = _draw_step_sizes(key, n_samples_prod, stage, scale)
    dimensionless_step_sizes = c * step_sizes
    return dimensionless_step_sizes, step_sizes, fitting_factor, c, state

def _sAIA_OptimalCoeffs(dimensionless_step_sizes, stage, key, n_coeff_samples = 256, n_h_samples = 256):
    """
    Compute optimal coefficients for s-AIA method
    -------------------------
    Parameters:
    dimensionless_step_sizes (jax.Array): dimensionless step sizes
    stage (int): number of stages
    key (int): random number generator key
    n_coeff_samples (int): number of coefficient samples
    -------------------------
    Returns:
    optimal_coeffs (jax.Array): optimal coefficient b for each dimensionless step size
    """
    if stage == 2:
        rho, b_ME, b_VV = _rho_2, ME_2().b, VV_2().b
    elif stage == 3:
        rho, b_ME, b_VV = _rho_3, ME_3().b, VV_3().b
    else:
        raise NotImplementedError("Only 2- & 3-stage integrators are supported as of now.")
    keys = jax.random.split(jax.random.PRNGKey(key), dimensionless_step_sizes.shape[0])
    f = jax.vmap(partial(_optimal_b, rho=rho, b_ME=b_ME, b_VV=b_VV,
                         n_coeff_samples=n_coeff_samples))
    return jax.jit(f)(dimensionless_step_sizes, keys)

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

_SEP = "=" * 61

def _print_header(title, rows):
    """
    Print the header table of an adaptive scheme.
    -------------------------
    Parameters:
        title (str): title line
        rows (list): (label, value) pairs, one table row each
    """
    print(title)
    print(_SEP)
    for label, value in rows:
        print(f"{label:^30}|{value:^30}")
    print(_SEP)

def _header_rows(sampler, n_samples_tune, n_samples_check, n_samples_burn_in, n_samples_prod,
                 stage, target_AR, sensibility, step_factor, compute_freqs):
    """Rows shared by the headers of ATune and sMAIA."""
    return [("Sampler", sampler),
            ("Num. Samples Tune", n_samples_tune),
            ("Num. Samples Check", n_samples_check),
            ("Num. Samples Burn-In", n_samples_burn_in),
            ("Num. Samples Prod", n_samples_prod),
            ("Stage", stage),
            ("Target AR", target_AR),
            ("Sensibility", sensibility),
            ("Step Factor", step_factor),
            ("Compute Freqs", "Yes" if compute_freqs else "No")]

def _summary(values):
    """Compact min, max, mean string of an array, for printing."""
    v = jnp.asarray(values)
    return f"min {float(v.min()):.4g}, max {float(v.max()):.4g}, mean {float(v.mean()):.4g}"

def _momentum_noise_bounds(stage, D):
    """
    Lower and upper bounds of the momentum noise of GHMC in ATune.
    -------------------------
    Parameters:
        stage (int): number of stages (2 or 3)
        D (int): dimension of data
    -------------------------
    Returns:
        (phi_lower, phi_upper): bounds, computed at the dimensionless step sizes stage and 2.0772
    """
    if stage == 2: # TODO: FIX momentum assignation for 2-stage
        a_low, b_low, a_up, b_up = 0, 1/4, 0, 1/4
    else:
        b_low, b_up = 0.11888010966548, 0.113252
        a_low = (1 - 2*b_low)/(4*(1 - 3*b_low))
        a_up = (1 - 2*b_up)/(4*(1 - 3*b_up))
    return (optimal_momentum_noise(stage, stage, D, a_low, b_low),
            optimal_momentum_noise(2.0772, stage, D, a_up, b_up))

# Number of integration steps used in the tuning and burn-in stages of ATune (always 1)
_L_TUNE_BURN_IN = 1
# Values of L drawn uniformly in production when L is random
_L_RANDOM_VALUES = (2, 5, 7)
_N_STEPS_MODES = ("fixed", "random")

def _production_n_steps(key, n, compute_hessian, fitting_factor, n_steps_mode):
    """
    Number of integration steps L for each production iteration of ATune.
    -------------------------
    Parameters:
        key (jax.Array): PRNG key
        n (int): number of production iterations
        compute_hessian (bool): whether the Hessian is computed
        fitting_factor (float): scaling factor of the step size interval, from the burn-in
        n_steps_mode (str): "fixed" (L = 1) or "random" (L uniform in {2, 5, 7}),
            used only when compute_hessian is False
    -------------------------
    Returns:
        n_steps (jax.Array): integer array of shape (n,)
    """
    random_L = (fitting_factor > 1.5) if compute_hessian else (n_steps_mode == "random")
    if random_L:
        return jax.random.choice(key, jnp.array(_L_RANDOM_VALUES, dtype=jnp.int32), shape=(n,))
    return jnp.ones(n, dtype=jnp.int32)

def ATune(x_init, potential_args, n_samples_tune, n_samples_check,
         n_samples_burn_in, n_samples_prod, potential, mass_matrix,
         target_AR = 0.92, stage = 2, sensibility = 0.01,
         step_factor = 0.5, compute_freqs = True, compute_hessian = True, sampler = "HMC",
         RNG_key = 42, initial_step_size=None, progress=True, n_steps_mode="fixed"):
    """
    ATune: automatic tuning of step size, number of steps, momentum noise and integrator (ex s-AIA)

    Note: As of this version the ATune method is only supported for 2- & 3-stage
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
        step_factor (float): step size relative increment/decrement
        compute_freqs (bool): compute frequencies
        compute_hessian (bool): compute hessian function
        sampler (str): sampler type
        RNG_key (int): random number generator key
        initial_step_size (float or None): if None, it's set to 1/D
        progress (bool): show the progress bar
        n_steps_mode (str): number of integration steps L in production when compute_hessian=False:
            "fixed" (default) uses L = 1 at every iteration,
            "random" draws L uniformly in {2, 5, 7} at every iteration.
            Ignored (with a warning) when compute_hessian=True, where L is chosen from the fitting factor.
            Tuning and burn-in always use L = 1.
    -------------------------
    Returns:
        samples (jax.Array): samples
    """
    #TODO: Extend functionality to other samplers and generalize to k-stages
    if stage not in [2, 3]:
        raise NotImplementedError("Only 2- & 3-stage integrators are supported as of now.")
    if sampler not in ["HMC", "GHMC"]:
        raise NotImplementedError("Only HMC & GHMC samplers are supported as of now.")
    if initial_step_size is not None and initial_step_size <= 0:
        raise ValueError("initial_step_size must be positive")
    if n_steps_mode not in _N_STEPS_MODES:
        raise ValueError(f"n_steps_mode must be one of {_N_STEPS_MODES}, got {n_steps_mode!r}")
    if compute_hessian and n_steps_mode != "fixed":
        warnings.warn("n_steps_mode is used only when compute_hessian=False: it is ignored, "
                      "and L is chosen from the fitting factor.", stacklevel=2)
    _print_header("Running ATune Adaptive Integration Scheme...",
                _header_rows(sampler, n_samples_tune, n_samples_check, n_samples_burn_in,
                            n_samples_prod, stage, target_AR, sensibility, step_factor, compute_freqs)
                + [("Compute Hessian", "Yes" if compute_hessian else "No")]
                + ([] if compute_hessian else [("Integration Steps Mode", n_steps_mode)]))
    potential = jax.jit(jax.tree_util.Partial(potential, *potential_args))
    potential_grad = jax.jit(jax.grad(potential))
    if compute_hessian == True: potential_hessian = jax.jit(jax.hessian(potential))
    else: potential_hessian = None 
    if sampler == "HMC":
        step_fn = make_hmc_step(potential, potential_grad, mass_matrix, None, None, None)
    elif sampler == "GHMC":
        step_fn = make_ghmc_step(potential, potential_grad, mass_matrix, None, None, None, None, True)

    # Step 1: Tuning Stage
    print("1) Tuning Stage...")
    step_size = (1.0 / x_init.shape[0]) if initial_step_size is None else float(initial_step_size)
    # Tuning and burn-in always use L = 1, whatever compute_hessian and n_steps_mode are
    n_steps = _L_TUNE_BURN_IN
    n_samples, integrator = n_samples_tune, VerletIntegrator()
    momentum_noise_lower, momentum_noise_upper = None, None
    if sampler == "GHMC":
        momentum_noise_lower, momentum_noise_upper = _momentum_noise_bounds(stage, x_init.shape[0])
    print(f"\t- Number of Tuning Samples: {n_samples}")
    print(f"\t- Dimension of Data: {x_init.shape[0]}")
    print(f"\t- Initial Step-Size: {step_size}")
    if sampler == "GHMC":
        print(f"\t- Initial Momentum Noise (Lower Bound): {momentum_noise_lower}")
        print(f"\t- Initial Momentum Noise (Upper Bound): {momentum_noise_upper}")
    tuned_step_size, AR_tuned, N_tune, state = _sAIA_Tuning(x_init, n_samples, n_samples_check, step_size, n_steps,
                                            sensibility, target_AR, mass_matrix, step_factor, integrator,
                                            sampler, momentum_noise_lower, momentum_noise_upper,
                                            jax.random.PRNGKey(RNG_key), step_fn, progress)
    converged = abs(AR_tuned - target_AR) <= sensibility
    print(f"\t- Tuning Iterations: {N_tune} of {n_samples_tune} ({'converged' if converged else 'budget exhausted'})")
    print(f"\t- Tuned AR: {AR_tuned}")
    print(f"\t- Tuned Step-Size: {tuned_step_size:.5f}")
    print(_SEP)

    # Step 2: Burn-In Stage
    print("2) Burn-In Stage...")
    n_samples, step_size = n_samples_burn_in, tuned_step_size
    print(f"\t- Number of Burn-In Samples: {n_samples}")
    dimensionless_step_sizes, step_sizes, fitting_factor, c, state = _sAIA_BurnIn(
            x_init, n_samples, n_samples_prod, compute_freqs, step_size, n_steps, stage,
            potential_hessian, mass_matrix, integrator, sampler, momentum_noise_lower,
            momentum_noise_upper, jax.random.PRNGKey(RNG_key), step_fn, progress, state0=state)
    print(f"\t- Dimensionless Step-Sizes: {_summary(dimensionless_step_sizes)}")
    print(f"\t- Step-Sizes: {_summary(step_sizes)}")
    print(_SEP)

    # Step 3: Production Stage
    print("3) Production Stage...")
    # n_steps = jax.random.randint(jax.random.PRNGKey(RNG_key), shape=(n_samples_prod,), minval=1, maxval=2 * (x_init.shape[0] / step_sizes) - 1)
    n_steps = _production_n_steps(jax.random.PRNGKey(RNG_key), n_samples_prod,
                                  compute_hessian, fitting_factor, n_steps_mode)
    print(f"\t- Number of Steps: {_summary(n_steps)}")
    keys = jax.random.split(jax.random.PRNGKey(RNG_key), n_samples_prod)
    cs = jnp.full((n_samples_prod,), c)
    if stage == 2:
        integrators = sAIA_2(c=cs, key=keys)
    else:
        integrators = sAIA_3(c=cs, key=keys)
    key = jax.random.PRNGKey(RNG_key)
    per_iter = _per_iter(sampler, n_samples_prod, step_sizes, n_steps, integrators,
                         key, momentum_noise_lower, momentum_noise_upper)
    samples, N_acc, _, _ = _sAIA_run(x_init, n_samples_prod, 100, per_iter, step_fn, sampler,
                                  mass_matrix, None, key, progress=True, state0=state)
    # Wait for the async computation to finish, so the prints below come after the progress bar
    jax.block_until_ready((samples, N_acc)) 
    AR_prod = acceptance_rate(N_acc, n_samples_prod)
    print(_SEP)
    print(f"Production stage finished, production acceptance rate: {AR_prod}.")
    return samples

def sAIA(*args, **kwargs):
    """
    Alias kept for backward compatibility: forwards to ATune.
    -------------------------
    Parameters:
        *args, **kwargs: forwarded unchanged to ATune
    -------------------------
    Returns:
        the output of ATune
    """
    return ATune(*args, **kwargs)

def _sMAIA_MHMC(x_init, n_samples, burn_in, per_iter, potential, potential_grad,
                potential_hessian, mass_matrix, order, key):
    """
    Single chain Modified Hamiltonian Monte Carlo (MHMC) sampler for s-MAIA, run with lax.scan.
    Handles adaptive step sizes, numbers of steps, integrators and momentum noise.
    -------------------------
    Parameters:
        x_init (jax.Array): initial position
        n_samples (int): number of samples
        burn_in (int): burn in iterations
        per_iter (tuple): per iteration inputs (h, L, phi, integrator), as built by _per_iter with sampler "GHMC"
        potential (function): Hamiltonian potential
        potential_grad (function): Hamiltonian potential gradient
        potential_hessian (function): Hamiltonian potential Hessian
        mass_matrix (jax.Array): mass matrix
        order (int): order of the Modified Hamiltonian (4 or 6)
        key (jax.Array): PRNG key
    -------------------------
    Returns:
        samples (jax.Array): samples after the burn in
        weights (jax.Array): importance sampling weights of the samples
        N_acc (int): number of accepted proposals after the burn in
    """
    # Initial momentum (gaussian), shape given by mass matrix
    p0 = as_mass(mass_matrix).sample(key)
    state0 = (x_init, p0, Position_Terms(x_init, potential, potential_grad, potential_hessian))
    step_fn = make_mhmc_step(potential, potential_grad, potential_hessian, mass_matrix, order)
    samples, weights, accepted = _run_chain(state0, key, n_samples + burn_in, step_fn,
                                            _pad_front(per_iter, burn_in), n_keys=3, progress=True)
    return samples[burn_in:], weights[burn_in:], accepted[burn_in:].sum()

def sMAIA(x_init, potential_args, n_samples_tune, n_samples_check, 
          n_samples_burn_in, n_samples_prod, potential, mass_matrix, 
          target_AR=0.92, stage=2, sensibility=0.01, 
          step_factor=0.5, compute_freqs=True, order=4, RNG_key=42,
          initial_step_size=None, progress=True):
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
        step_factor (float): step size increment/decrement
        compute_freqs (bool): compute frequencies for adaptive tuning
        order (int): order of the Modified Hamiltonian (default is 4)
        RNG_key (int): random number generator key
        initial_step_size (float or None): if None, it's set to 1/D
        progress (bool): show the progress bar
    -------------------------
    Returns:
        samples (jax.Array): samples from the MHMC sampler
    """
    if initial_step_size is not None and initial_step_size <= 0:
        raise ValueError("initial_step_size must be positive")
    _print_header("Running s-MAIA (s-AIA for MHMC) Adaptive Integration Scheme...",
                _header_rows("MHMC", n_samples_tune, n_samples_check, n_samples_burn_in,
                            n_samples_prod, stage, target_AR, sensibility, step_factor, compute_freqs)
                + [("Order of Modified Hamiltonian", order)])

    # JIT compile potential and its derivatives
    potential = jax.jit(jax.tree_util.Partial(potential, *potential_args))
    potential_grad = jax.jit(jax.grad(potential))
    potential_hessian = jax.jit(jax.hessian(potential))
    key = jax.random.PRNGKey(RNG_key)
    step_fn = make_ghmc_step(potential, potential_grad, mass_matrix, None, None, None, None, True)

    # === 1. Tuning Stage ===
    print("1) Tuning Stage...")
    # Use GHMC for efficient initial tuning
    step_size = (1.0 / x_init.shape[0]) if initial_step_size is None else float(initial_step_size)
    n_steps = 1
    if stage == 2: a_tune, b_tune = 0, 1/4
    elif stage == 3: a_tune, b_tune = 1/3, 1/6
    print(f"\t- Number of Tuning Samples: {n_samples_tune}")
    print(f"\t- Dimension of Data: {x_init.shape[0]}")
    print(f"\t- Initial Step-Size: {step_size}")
    
    momentum_noise_lower = optimal_momentum_noise(stage, stage, x_init.shape[0], a_tune, b_tune)
    momentum_noise_upper = optimal_momentum_noise(1.5 if stage == 2 else 2.0772, stage, x_init.shape[0], a_tune, b_tune)
    
    print(f"\t- Initial Momentum Noise (Lower Bound): {momentum_noise_lower}")
    print(f"\t- Initial Momentum Noise (Upper Bound): {momentum_noise_upper}")
    # Tune the step size using the adaptive tuning function
    tuned_step_size, AR_tuned, N_tune, state = _sAIA_Tuning(
        x_init, n_samples_tune, n_samples_check, step_size, n_steps, sensibility,
        target_AR, mass_matrix, step_factor, VerletIntegrator(), "GHMC",
        momentum_noise_lower, momentum_noise_upper, key, step_fn, progress)
    converged = abs(AR_tuned - target_AR) <= sensibility
    print(f"\t- Tuning Iterations: {N_tune} of {n_samples_tune} ({'converged' if converged else 'budget exhausted'})")
    print(f"\t- Tuned AR: {AR_tuned}")
    print(f"\t- Tuned Step-Size: {tuned_step_size}")
    print(_SEP)
    # === 2. Burn-In Stage ===
    print("2) Burn-In Stage...")
    # Run MHMC to estimate frequencies with the tuned step size
    key, _ = jax.random.split(key)
    burn_in_integrator = VerletIntegrator()
    # Use frequency information to compute adaptive parameters for the production run
    dimensionless_step_sizes, step_sizes, _, _, state = _sAIA_BurnIn(
        x_init, n_samples_burn_in, n_samples_prod, compute_freqs, tuned_step_size, n_steps,
        stage, potential_hessian, mass_matrix, burn_in_integrator, "GHMC",
        momentum_noise_lower, momentum_noise_upper, key, step_fn, progress=progress, state0=state)
    
    opt_integration_coeffs = _sAIA_OptimalCoeffs(dimensionless_step_sizes, stage, RNG_key)
    print("\t- Dimensionless Step-Sizes computed.")
    print("\t- Optimal Integration Coefficients computed.")
    print(_SEP)

    # === 3. Production Stage ===
    print("3) Production Stage...")
    key, subkey1, subkey2 = jax.random.split(key, 3)
    
    # Adaptive integrators, number of steps and momentum noise for the production run
    prod_steps = jax.random.randint(subkey1, shape=(n_samples_prod,), minval=1, maxval=jnp.maximum(2, 2 * (x_init.shape[0] / step_sizes) - 1))

    if stage == 2:
        prod_integrators = MSSI_2(opt_integration_coeffs)
    else: # stage == 3
        a_coeffs = (2 * opt_integration_coeffs - 1) / (2 * (6 * opt_integration_coeffs - 2))
        prod_integrators = MSSI_3(a_coeffs, opt_integration_coeffs)
    per_iter = _per_iter("GHMC", n_samples_prod, step_sizes, prod_steps, prod_integrators,
                         subkey2, momentum_noise_lower, momentum_noise_upper)

    # Run the final production sampling with all adaptive parameters
    samples, weights, _ = _sMAIA_MHMC(
        state[0], n_samples_prod, 100, per_iter,
        potential, potential_grad, potential_hessian, mass_matrix, order, key)
    # Wait for the async computation to finish, so the prints below come after the progress bar
    jax.block_until_ready((samples, weights))
    print(_SEP)
    print("s-MAIA sampling complete.")
    
    return samples, weights
