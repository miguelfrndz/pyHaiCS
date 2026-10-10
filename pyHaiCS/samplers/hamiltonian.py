import jax
import jax.numpy as jnp
from tqdm import tqdm
from jax_tqdm import scan_tqdm
from ..integrators.integrators import VerletIntegrator, RMHMCVerletIntegrator, get_mhmc_coeffs
from ..utils.hamiltonian import Hamiltonian, Hamiltonian_RMHMC, Modified_Hamiltonian, Extended_Hamiltonian, Kinetic, Position_Terms, Modified_Hamiltonian_From_Terms, as_mass
from functools import partial

@partial(jax.jit, static_argnames=("n_total", "step_fn", "n_keys", "progress", "return_state"))
def _run_chain(state0, key, n_total, step_fn, xs=None, n_keys=2, progress=False, return_state=False):
    """
    Run a chain of n_total iterations with jax.lax.scan.
    -------------------------
    Parameters:
    state0 (pytree): initial state of the chain (the position x, or the pair (x, p))
    key (jax.Array): PRNG key
    n_total (int): number of iterations (static, fixed at compile time)
    step_fn (function): one iteration, called as step_fn(state, k1, k2, per_iter), returns (new_state, output).
    xs (pytree or None): values that change at every iteration, with leading axis n_total (step sizes, number of steps, ..), passed to step_fn as per_iter
    n_keys (int): number of PRNG keys given to step_fn at every iteration (static);
    progress (bool): show a tqdm progress bar with about 100 updates (static)
    return_state (bool): return the final state (static)
    -------------------------
    Returns:
    outputs (pytree): the outputs of step_fn, stacked along a new leading axis of length n_total
    state (pytree): only if return_state is True, the final state of the chain, returned as (state, outputs)
    """
    def body(carry, inp):
        _, per_iter = inp
        state, key = carry
        key, *ks = jax.random.split(key, n_keys + 1)
        state, out = step_fn(state, *ks, per_iter)
        return (state, key), out
    if progress:
        body = scan_tqdm(n_total, print_rate=max(1, n_total // 100))(body)
    (state_f, _), outputs = jax.lax.scan(body, (state0, key),
                                   (jnp.arange(n_total), xs), length=n_total)
    return (state_f, outputs) if return_state else outputs

def make_hmc_step(potential, potential_grad, mass_matrix, integrator, step_size, n_steps):
    """
    Build the function that performs one HMC iteration, to be used by _run_chain.
    -------------------------
    Parameters:
        potential (function): Hamiltonian potential
        potential_grad (function): Hamiltonian potential gradient
        mass_matrix (jax.Array): mass matrix
        integrator (object): default integrator, used when per_iter is None
        step_size (float): default step size, used when per_iter is None
        n_steps (int): default number of integration steps, used when per_iter is None
    -------------------------
    Returns:
        step_fn (function): step_fn(x, k_mom, k_acc, per_iter) returns (x_new, (x_new, accept)). per_iter is None for fixed parameters, otherwise the triple (h, L, integrator)
    """
    mass = as_mass(mass_matrix)
    def step_fn(x, k_mom, k_acc, per_iter):
        h, L, integ = (step_size, n_steps, integrator) if per_iter is None else per_iter
        p = mass.sample(k_mom)
        x_prop, p_prop = integ.integrate(x, p, potential_grad, L, mass, h)
        delta_H = Hamiltonian(x_prop, p_prop, potential, mass) - Hamiltonian(x, p, potential, mass)
        accept = jax.random.uniform(k_acc) < jnp.exp(-delta_H)
        x_new = jnp.where(accept, x_prop, x)
        return x_new, (x_new, accept)
    return step_fn

def _single_chain_HMC(x_init, n_samples, burn_in, step_size, n_steps,
        potential, potential_grad, mass_matrix, integrator, key):
    """
    Single-Chain Hamiltonian Monte-Carlo (HMC) sampler.
    -------------------------
    Parameters:
        x_init (jax.Array): initial position
        n_samples (int): number of samples
        burn_in (int): burn-in samples
        step_size (float): step size
        n_steps (int): number of integration steps
        potential (function): Hamiltonian potential
        potential_grad (function): Hamiltonian potential gradient
        mass_matrix (jax.Array): mass matrix
        integrator (object): integrator object
        key (jax.Array): PRNG key
    -------------------------
    Returns:
        samples (jax.Array): samples after the burn-in
    """
    step_fn = make_hmc_step(potential, potential_grad, mass_matrix, integrator, step_size, n_steps)
    samples, _ = _run_chain(x_init, key, n_samples + burn_in, step_fn, progress=True)
    return samples[burn_in:]

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
    print(f"{'Integrator':^30}|{integrator.name:^30}")
    print(f"{'Num. Integration Steps':^30}|{n_steps:^30}")
    print("="*61)
    keys = jax.random.split(jax.random.PRNGKey(RNG_key), n_chains)
    x_init_repeated = jnp.repeat(x_init[None, :], n_chains, axis = 0)
    potential = jax.jit(jax.tree_util.Partial(potential, *potential_args))
    potential_grad = jax.jit(jax.grad(potential))
    vectorized_chain = jax.vmap(_single_chain_HMC, in_axes=(0, None, None, None, None, None, None, None, None, 0))
    samples = vectorized_chain(x_init_repeated, n_samples, burn_in, step_size, n_steps, potential, potential_grad, mass_matrix, integrator, keys)
    return samples

def make_ghmc_step(potential, potential_grad, mass_matrix, integrator,
                   step_size, n_steps, momentum_noise, metropolize):
    """
    Build the function that performs one GHMC iteration, to be used by _run_chain.
    -------------------------
    Parameters:
        potential (function): Hamiltonian potential
        potential_grad (function): Hamiltonian potential gradient
        mass_matrix (jax.Array): mass matrix
        integrator (object): default integrator, used when per_iter is None
        step_size (float): default step size, used when per_iter is None
        n_steps (int): default number of integration steps, used when per_iter is None
        momentum_noise (float): default momentum noise, used when per_iter is None
        metropolize (bool): apply the Metropolis acceptance step (a Python bool, decided at compile time)
    -------------------------
    Returns:
        step_fn (function): step_fn(state, k_mu, k_acc, per_iter) with state = (x, p), returns ((x_out, p_out), (x_out, accept)). per_iter is None for fixed parameters, otherwise the tuple (h, L, phi, integrator)
    """
    mass = as_mass(mass_matrix)
    def step_fn(state, k_mu, k_acc, per_iter):
        """One GHMC iteration: partial momentum refresh, integrate, accept or reject."""
        x, p = state
        h, L, phi, integ = ((step_size, n_steps, momentum_noise, integrator)
                            if per_iter is None else per_iter)
        mu = mass.sample(k_mu)
        p_prop = jnp.sqrt(1 - phi) * p + jnp.sqrt(phi) * mu
        x_new, p_new = integ.integrate(x, p_prop, potential_grad, L, mass, h)
        if metropolize:                       
            delta_H = (Hamiltonian(x_new, p_new, potential, mass)
                       - Hamiltonian(x, p_prop, potential, mass))
            accept = jax.random.uniform(k_acc) < jnp.exp(-delta_H)
            x_out = jnp.where(accept, x_new, x)
            p_out = jnp.where(accept, p_new, -p_prop)      # momentum flip
        else:
            accept, x_out, p_out = jnp.array(True), x_new, p_new
        return (x_out, p_out), (x_out, accept)
    return step_fn

def make_mhmc_step(potential, potential_grad, potential_hessian, mass_matrix, order):
    """
    Build the function that performs one MHMC iteration with adaptive parameters, to be used by _run_chain.
    The derivatives of the potential at the current position (U, gradient, Hessian) are carried in the state:
    only one Hessian per iteration is computed, at the proposed position.
    -------------------------
    Parameters:
        potential (function): Hamiltonian potential
        potential_grad (function): Hamiltonian potential gradient
        potential_hessian (function): Hamiltonian potential Hessian
        mass_matrix (jax.Array): mass matrix
        order (int): order of the Modified Hamiltonian (4 or 6)
    -------------------------
    Returns:
        step_fn (function): step_fn(state, k_mu, k_mom, k_pos, per_iter) with state = (x, p, terms), where terms = Position_Terms(x, ...),
        returns ((x_out, p_out, terms_out), (x_out, weight, accept)). per_iter is the tuple (h, L, phi, integrator)
    """
    mass = as_mass(mass_matrix)

    def step_fn(state, k_mu, k_mom, k_pos, per_iter):
        """One MHMC iteration: partial momentum refresh with its own acceptance, integrate, accept or reject."""
        x, p, terms = state
        h, L, phi, integ = per_iter
        c = get_mhmc_coeffs(order=order, stage=integ.stage, b=integ.b, a=getattr(integ, 'a', None))

        def mod_H(x_, p_, terms_):
            return Modified_Hamiltonian_From_Terms(x_, p_, terms_, potential_grad, mass, h, order, c, mass.Minv)

        # Partial momentum refresh and its Metropolis step, on the extended Hamiltonian
        mu = mass.sample(k_mu)
        p_prop = jnp.sqrt(1 - phi) * p + jnp.sqrt(phi) * mu
        mu_prop = -jnp.sqrt(phi) * p + jnp.sqrt(1 - phi) * mu
        H_old, H_prop = mod_H(x, p, terms), mod_H(x, p_prop, terms)
        dH_ext = Extended_Hamiltonian(H_prop, mu_prop, mass) - Extended_Hamiltonian(H_old, mu, mass)
        accept_momentum = jax.random.uniform(k_mom) < jnp.exp(-dH_ext)
        p_accepted = jnp.where(accept_momentum, p_prop, p)
        H_initial = jnp.where(accept_momentum, H_prop, H_old)
        # Integrate, then Metropolis step on the position
        x_new, p_new = integ.integrate(x, p_accepted, potential_grad, L, mass, h)
        terms_new = Position_Terms(x_new, potential, potential_grad, potential_hessian)
        dH = mod_H(x_new, p_new, terms_new) - H_initial
        accept = jax.random.uniform(k_pos) < jnp.exp(-dH)
        x_out = jnp.where(accept, x_new, x)
        p_out = jnp.where(accept, p_new, -p_accepted)                     # momentum flip
        terms_out = jax.tree_util.tree_map(lambda a, b: jnp.where(accept, a, b), terms_new, terms)
        # Importance sampling weight, reusing the terms of the output position
        weight = jnp.exp(terms_out[0] + Kinetic(p_out, mass) - mod_H(x_out, p_out, terms_out))
        return (x_out, p_out, terms_out), (x_out, weight, accept)
    return step_fn

def _single_chain_GHMC(x_init, n_samples, burn_in, step_size, n_steps, potential, potential_grad,
        mass_matrix, momentum_noise, integrator, key, metropolize):
    """
    Single-Chain Generalized Hamiltonian Monte-Carlo (GHMC) sampler.
    -------------------------
    Parameters:
        x_init (jax.Array): initial position
        n_samples (int): number of samples
        burn_in (int): burn-in samples
        step_size (float): step size
        n_steps (int): number of integration steps
        potential (function): Hamiltonian potential
        potential_grad (function): Hamiltonian potential gradient
        mass_matrix (jax.Array): mass matrix
        momentum_noise (float): momentum noise
        integrator (object): integrator object
        key (jax.Array): PRNG key
        metropolize (bool): apply the Metropolis acceptance step
    -------------------------
    Returns:
        samples (jax.Array): samples after the burn-in
    """
    key, k0 = jax.random.split(key)
    p0 = as_mass(mass_matrix).sample(k0)
    step_fn = make_ghmc_step(potential, potential_grad, mass_matrix, integrator,
                             step_size, n_steps, momentum_noise, metropolize)
    samples, _ = _run_chain((x_init, p0), key, n_samples + burn_in, step_fn, progress=True)
    return samples[burn_in:]

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
    print(f"{'Integrator':^30}|{integrator.name:^30}")
    print(f"{'Num. Integration Steps':^30}|{n_steps:^30}")
    print(f"{'Momentum Noise':^30}|{momentum_noise:^30}")
    print("="*61)
    keys = jax.random.split(jax.random.PRNGKey(RNG_key), n_chains)
    x_init_repeated = jnp.repeat(x_init[None, :], n_chains, axis = 0)
    potential = jax.tree_util.Partial(potential, *potential_args)
    potential_grad = jax.grad(potential)
    vectorized_chain = jax.vmap(_single_chain_GHMC, in_axes=(0, None, None, None, None, None, None, None, None, None, 0, None))
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

def _single_chain_MHMC(x_init, n_samples, burn_in, step_size, n_steps, 
        potential, potential_grad, potential_hessian, mass_matrix, momentum_noise, integrator, key, order, c):
    """
    Single-chain Modified Hamiltonian Monte Carlo (MHMC) sampling.
    """
    samples = []
    weights = []
    x = x_init
    mass = as_mass(mass_matrix)
    # Initial momentum (gaussian), shape given by mass matrix
    p = mass.sample(key)
    for n in tqdm(range(n_samples + burn_in)):
        key, subkey = jax.random.split(key)
        # Sample noise vector
        mu = mass.sample(subkey)
        # Propose updated momentum and noise vector
        p_prop = jnp.sqrt(1 - momentum_noise) * p + jnp.sqrt(momentum_noise) * mu
        mu_prop = -jnp.sqrt(momentum_noise) * p + jnp.sqrt(1 - momentum_noise) * mu
        # Compute Modified Hamiltonians
        H_old_full = Modified_Hamiltonian(x, p, potential, potential_grad, potential_hessian, mass, step_size, order, c)
        H_prop_full = Modified_Hamiltonian(x, p_prop, potential, potential_grad, potential_hessian, mass, step_size, order, c)
        # Compute Difference of Extended Hamiltonians
        dH_ext = Extended_Hamiltonian(H_prop_full, mu_prop, mass) - Extended_Hamiltonian(H_old_full, mu, mass)
        # Metropolis-Hastings acceptance (for momentum)
        accept = jax.random.uniform(subkey) < jnp.exp(-dH_ext)
        p_accepted = jax.lax.cond(accept, lambda _: p_prop, lambda _: p, operand = None)
        # Integrate Hamiltonian dynamics with accepted momentum
        x_new, p_new = integrator.integrate(x, p_accepted, potential_grad, n_steps, mass, step_size)
        # Compute Difference of Modified Hamiltonians
        H_prop = Modified_Hamiltonian(x, p_accepted, potential, potential_grad, potential_hessian, mass, step_size, order, c)
        H_new = Modified_Hamiltonian(x_new, p_new, potential, potential_grad, potential_hessian, mass, step_size, order, c)
        dH = H_new - H_prop
        # Metropolis-Hastings acceptance (for position)
        accept2 = jax.random.uniform(subkey) < jnp.exp(-dH)
        x, p = jax.lax.cond(accept2, lambda _:(x_new, p_new), lambda _:(x, -p_accepted), operand=None)
        if n >= burn_in:
            samples.append(x)
            # Compute importance sampling weights
            weight = jnp.exp(Hamiltonian(x, p, potential, mass) - Modified_Hamiltonian(x, p, potential, potential_grad, potential_hessian, mass, step_size, order, c))
            weights.append(weight)
    samples = jnp.stack(samples, axis = 0)
    weights = jnp.stack(weights)
    return samples, weights

def MHMC(x_init, potential_args, n_samples, burn_in, step_size, n_steps, 
        potential, mass_matrix, momentum_noise, integrator= VerletIntegrator(), 
        order = 4, n_chains = 4, RNG_key = 42):
    """ 
    Multi-Chain Modified Hamiltonian Monte Carlo (MHMC) sampler.
    """
    print(f"Running MHMC-{order} sampler...")
    print("="*61)
    print(f"{'Num. Chains':^30}|{n_chains:^30}")
    print(f"{'Num. Samples':^30}|{n_samples:^30}")
    print(f"{'Num. Burn-In Iterations':^30}|{burn_in:^30}")
    print(f"{'Step-Size':^30}|{step_size:^30}")
    print(f"{'Integrator':^30}|{integrator.name:^30}")
    print(f"{'Num. Integration Steps':^30}|{n_steps:^30}")
    print(f"{'Momentum Noise':^30}|{momentum_noise:^30}")
    print(f"{'Order of Modified Hamiltonian':^30}|{order:^30}")
    print("="*61)

    keys = jax.random.split(jax.random.PRNGKey(RNG_key), n_chains)
    x_init_repeated = jnp.repeat(x_init[None, :], n_chains, axis = 0)
    potential = jax.tree_util.Partial(potential, *potential_args)
    potential_grad = jax.jit(jax.grad(potential))
    potential_hessian = jax.jit(jax.hessian(potential))
    coeffs = get_mhmc_coeffs(order = order, stage = integrator.stage, 
                             b = integrator.b, a = getattr(integrator, 'a', None))

    vectorized_chain = jax.vmap(
        _single_chain_MHMC,
        in_axes=(0, None, None, None, None, None, None, None, None, None, None, 0, None, None)
    )

    samples, weights = vectorized_chain(
        x_init_repeated, n_samples, burn_in, step_size, n_steps,
        potential, potential_grad, potential_hessian, mass_matrix,
        momentum_noise, integrator, keys, order, coeffs
    )
    return samples, weights

def _single_chain_RMHMC(x_init, n_samples, burn_in, step_size, n_steps,
                        potential, metric, integrator, key):
    """
    Single-Chain Riemannian Manifold Hamiltonian Monte Carlo (RMHMC).
    -------------------------
    Parameters:
        x_init (jax.Array): initial position
        n_samples (int): number of samples
        burn_in (int): burn-in samples
        step_size (float): step size
        n_steps (int): number of integration steps
        potential (function): Hamiltonian potential
        metric (function): metric tensor function G(theta)
        integrator (object): integrator with integrate_rmhmc()
    -------------------------
    Returns:
        samples (jax.Array): samples
    """
    samples = []
    hamiltonian = jax.tree_util.Partial(Hamiltonian_RMHMC, potential=potential, metric=metric)
    x = x_init
    for n in tqdm(range(n_samples + burn_in)):
        key, subkey = jax.random.split(key)
        G = metric(x)
        p = jax.random.multivariate_normal(subkey, jnp.zeros(x.shape[0]), G)
        x_prop, p_prop = integrator.integrate(x, p, hamiltonian, n_steps, step_size)
        # RMHMC Hamiltonian (non-separable)
        delta_H = hamiltonian(x_prop, p_prop) - hamiltonian(x, p)
        accept = jax.random.uniform(subkey) < jnp.exp(-delta_H)
        x = jax.lax.cond(accept, lambda _: x_prop, lambda _: x, operand=None)
        if n >= burn_in:
            samples.append(x)
    samples = jnp.stack(samples, axis=0)
    return samples

def RMHMC(x_init, potential_args, n_samples, burn_in, step_size, n_steps, potential, metric, integrator = RMHMCVerletIntegrator(), n_chains=4, RNG_key=42):
    """
    Multi-Chain Riemannian Manifold Hamiltonian Monte Carlo (RMHMC) sampler.
    -------------------------
    Parameters:
        x_init (jax.Array): initial position
        potential_args (tuple): arguments for Hamiltonian potential
        n_samples (int): number of samples
        burn_in (int): burn-in samples
        step_size (float): step-size
        n_steps (int): number of integration steps
        potential (function): Hamiltonian potential
        metric (function): metric tensor G(theta)
        integrator (object): integrator with integrate_rmhmc()
        n_chains (int): number of chains
        RNG_key (int): random seed
    -------------------------
    Returns:
        samples (jax.Array): samples
    """
    print(f"Running RMHMC sampler...")
    print("=" * 61)
    print(f"{'Num. Chains':^30}|{n_chains:^30}")
    print(f"{'Num. Samples':^30}|{n_samples:^30}")
    print(f"{'Num. Burn-In Iterations':^30}|{burn_in:^30}")
    print(f"{'Step-Size':^30}|{step_size:^30}")
    print(f"{'Integrator':^30}|{integrator.name:^30}")
    print(f"{'Num. Integration Steps':^30}|{n_steps:^30}")
    print("=" * 61)
    keys = jax.random.split(jax.random.PRNGKey(RNG_key), n_chains)
    x_init_repeated = jnp.repeat(x_init[None, :], n_chains, axis=0)
    potential = jax.tree_util.Partial(potential, *potential_args)
    # metric = jax.tree_util.Partial(metric, *potential_args)
    vectorized_chain = jax.vmap(_single_chain_RMHMC, in_axes=(0, None, None, None, None, None, None, None, 0))
    samples = vectorized_chain(x_init_repeated, n_samples, burn_in, step_size, n_steps, potential, metric, integrator, keys)
    return samples
