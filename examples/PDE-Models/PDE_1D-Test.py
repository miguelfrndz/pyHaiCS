import sys, os
from pathlib import Path
sys.path.append('../')
HERE = Path(__file__).resolve().parent        # examples/PDE-Models
ROOT = HERE.parents[1]                        # repo root, pyHaiCS-dev

import numpy as np
from model_1D import Model1D
import matplotlib.pyplot as plt
import time
from pyHaiCS.integrators.integrators import VerletIntegrator, BCSS_3, VV_2
from pyHaiCS.samplers.basic_mcmc import _single_chain_RWMH, RWMH
from pyHaiCS.samplers.hamiltonian import _single_chain_HMC, HMC, _single_chain_GHMC
from pyHaiCS.samplers.adaptive import ATune
import jax
import jax.numpy as jnp
 #jax.config.update('jax_disable_jit', True)
jax.config.update("jax_enable_x64", True)
from jax.scipy.special import gammaln
from functools import partial
import equinox as eqx

# Enable this for debugging (for finding NaNs):
jax.config.update("jax_debug_nans", False)

PLOT = False

D = 0.48
m = 3
r = 0
K = 1
sigma = 0.5
aw = 2

L = 6
N = 50
dx = 2*L/N

nsteps = 100
dt = 5/nsteps
tspan = jnp.arange(0, nsteps+1) * dt

@eqx.filter_jit
def gaussian(x, x0, sigma):
  return  1./(jnp.sqrt(2.*jnp.pi)) * jnp.exp(-jnp.power((x - x0)/sigma, 2.)/2.)

x = jnp.linspace(-L,L,N)

u0 = gaussian(x, 1, 0.5) + gaussian(x,-1,0.5)

model = Model1D(
    u0_shape=u0.shape, L=L, N=N, dt=dt, nsteps=nsteps,
    D=D, r=r, K=K, m=m, sigma=sigma, aw=aw
)

start = time.time()
times, solution = model.solve(u0)
end = time.time()
print(f'Time to solve ODE: {end-start:.4f} seconds')

if PLOT:
    print("Plotting results...")
    # Plot the final state
    plt.figure(figsize=(8, 5))
    plt.plot(model.x, solution[-1], label=f'State at t={times[-1]:.2f}')
    plt.title("Final State of the System")
    plt.xlabel("Position x")
    plt.ylabel("Density u")
    plt.grid(True)
    plt.legend()
    plt.show()

    # Create the 3D surface plot
    fig, ax = plt.subplots(figsize=(10, 7), subplot_kw={"projection": "3d"})

    # Use the grid and time vector returned by the solver for the meshgrid
    X, Y = jnp.meshgrid(model.x, times)

    # Plot the surface
    ax.plot_surface(X, Y, solution, cmap=plt.cm.viridis, edgecolor='none', alpha=0.9)
    ax.set_title("System Evolution u(x,t)")
    ax.set_xlabel("Position x")
    ax.set_ylabel("Time t")
    ax.set_zlabel("Density u")
    ax.view_init(elev=30, azim=-60) # Adjust view angle
    plt.show()

    # fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    # X, Y = jnp.meshgrid(model.x, times)
    # # Plot the surface.
    # ax.plot_surface(X, Y, solution[:], edgecolor='royalblue', lw=0.5, rstride=8, cstride=8,
    #                 alpha=0.3)
    # ax.zaxis.set_major_locator(LinearLocator(10))

############# pyHaiCS Sampling Example #############

dataset = jnp.array(np.genfromtxt(ROOT / "pyHaiCS" / "benchmarks" / "PDE" / "1D_noisy_simulation.txt"))

@eqx.filter_jit
def log_exponential_dist(beta, x):
    return jnp.log(beta) - beta * x

@eqx.filter_jit
def log_gamma_dist(theta, alpha, x):
    return -gammaln(alpha) - alpha * jnp.log(theta) + (alpha - 1) * jnp.log(x) - (x/theta)

@eqx.filter_jit
def log_prior(x):
    beta_D = 1.0
    alpha_m = 10.0
    theta_m = 3.0
    beta_sigma_likelihood = 1.0

    D = x[0]
    m = x[1]
    sigma_likelihood = x[2]
    
    return log_exponential_dist(beta_D, D) + log_gamma_dist(theta_m, alpha_m, m) + log_exponential_dist(beta_sigma_likelihood, sigma_likelihood)

#@partial(jax.jit, static_argnames=('L', 'N', 'nsteps', 'dt', 'r', 'K', 'sigma', 'aw'))
@eqx.filter_jit
def log_likelihood(params, L, N, nsteps, dt, r, K, sigma, aw):
    D, m, sigma_likelihood = params[0], params[1], params[2]
    # Define initial state
    x_grid = jnp.linspace(-L, L, N)
    u0 = gaussian(x_grid, 1.0, 0.5) + gaussian(x_grid, -1.0, 0.5)
    # Instantiate the model
    model = Model1D(u0_shape = u0.shape, D = D, m = m, L = L, N = N, 
                    dt = dt, nsteps = nsteps, r = r, K = K, sigma = sigma, aw = aw)
    # Run solver
    _, solution = model.solve(u0)
    
    # Calculate log-likelihood
    num_data_points = (nsteps + 1) * N
    logL = -0.5 * jnp.sum(((dataset - solution) / sigma_likelihood)**2)
    logL -= num_data_points * (jnp.log(sigma_likelihood) + 0.5 * jnp.log(2 * jnp.pi))
    
    return logL

L_fixed, N_fixed, nsteps_fixed = 6.0, 50, 50
dt_fixed = 5.0 / nsteps_fixed
r_fixed, K_fixed, sigma_fixed, aw_fixed = 0.0, 1.0, 0.5, 2.0

@eqx.filter_jit
def potential(z):
    """Potential in the log parametrization z = log(D, m, sigma_likelihood)."""
    x = jnp.exp(z)
    logL = log_likelihood(x, L_fixed, N_fixed, nsteps_fixed, dt_fixed, r_fixed, K_fixed, sigma_fixed, aw_fixed)
    logP = log_prior(x)
    return -logL - logP - jnp.sum(z)

# @jax.jit
# def gradient_potential(x):
#     delta_D = 1e-4
#     delta_m = 1e-4

#     D = x[0]
#     m = x[1]
#     sigma_likelihood = x[2]

#     N = 50
#     nsteps = 50
    
#     post = potential(x)
#     post_Dplush = potential([D+delta_D, m, sigma_likelihood])
#     post_mplush = potential([D, m+delta_m, sigma_likelihood])

#     grad_D = (post_Dplush - post)/delta_D
#     grad_m = (post_mplush - post)/delta_m

#     #Expresión analitica del gradiente del potencial para sigma_likelihood 
#     grad_sigma_potential = nsteps*N/sigma_likelihood - jnp.sum((dataset - solution)**2/sigma_likelihood**3) + sigma_likelihood

#     return  jax.numpy.array([grad_D, grad_m, grad_sigma_potential])

# @jax.jit
# def manual_gradient_potential(x):
#     """
#     A corrected manual gradient calculator for testing purposes.
#     - D and m gradients are calculated with finite differences.
#     - sigma_likelihood gradient is calculated analytically.
#     """
#     # --- Parameters needed for the simulation ---
#     delta_D = 1e-5  # Using a smaller delta for better accuracy
#     delta_m = 1e-5
    
#     # Unpack parameters
#     D, m, sigma_likelihood = x[0], x[1], x[2]

#     # --- 1. Recalculate the solution to bring it into scope ---
#     # This is the key fix. We run the forward model once to get the `solution`
#     # array needed for the analytical part of the gradient.
#     # We use the same fixed parameters as in the `potential` function.
#     model = Model1D(u0_shape = (N_fixed,), D=D, m=m, L=L_fixed, N=N_fixed, dt=dt_fixed, nsteps=nsteps_fixed, 
#                     r=r_fixed, K=K_fixed, sigma=sigma_fixed, aw=aw_fixed)
#     x_grid = jnp.linspace(-L_fixed, L_fixed, N_fixed)
#     u0 = gaussian(x_grid, 1.0, 0.5) + gaussian(x_grid, -1.0, 0.5)
#     _, solution = model.solve(u0)

#     # --- 2. Calculate finite differences for D and m ---
#     # We call the `potential` function, which runs the simulation internally.
#     # Note: Using jnp.array() for the inputs is better practice than Python lists.
#     post = potential(x)
#     post_D_plus_h = potential(jnp.array([D + delta_D, m, sigma_likelihood]))
#     post_m_plus_h = potential(jnp.array([D, m + delta_m, sigma_likelihood]))

#     grad_D = (post_D_plus_h - post) / delta_D
#     grad_m = (post_m_plus_h - post) / delta_m

#     # --- 3. Calculate the analytical part for sigma_likelihood ---
#     # The `solution` variable is now available.
#     # Note: The gradient of the potential (-log(post)) is the negative
#     # of the gradient of the log-prior plus the log-likelihood.
#     # d/ds (-logL) = N_total/s - sum(err^2)/s^3
#     # d/ds (-logP) = beta
#     # So the original expression was missing the prior contribution.
    
#     # Gradient of -log(Likelihood) w.r.t sigma_likelihood
#     num_data_points = (nsteps_fixed + 1) * N_fixed
#     grad_L_sigma = (num_data_points / sigma_likelihood - 
#                     jnp.sum((dataset - solution)**2) / sigma_likelihood**3)

#     # Gradient of -log(Prior) w.r.t sigma_likelihood
#     beta_sigma_likelihood = 1.0 # From your log_prior function
#     grad_P_sigma = beta_sigma_likelihood

#     grad_sigma_potential = grad_L_sigma + grad_P_sigma

#     return jnp.array([grad_D, grad_m, grad_sigma_potential])

gradient_potential =eqx.filter_jit(eqx.filter_grad(potential))

params_init = jnp.log(jnp.array([1.0, 2.0, 1.0]))

print("Calculating initial potential...")
initial_potential = potential(params_init)
print(f"\t- Potential at initial parameters: {initial_potential:.4f}")

print("Calculating gradient at initial parameters...")
initial_gradient = gradient_potential(params_init)
print(f"\t- Gradient (D, m, sigma_L) at initial parameters: {initial_gradient}")

"""samples_GHMC = _single_chain_GHMC(params_init, 
                                n_samples = 1000, 
                                burn_in = 1000, 
                                step_size = 0.001, 
                                n_steps = 1, 
                                potential = potential, 
                                potential_grad = gradient_potential,
                                mass_matrix = jax.numpy.identity(3),
                                integrator = BCSS_3(),
                                momentum_noise = 0.5,
                                metropolize=False,
                                key = jax.random.PRNGKey(120))"""


samples_GHMC = ATune(x_init =params_init, 
                    potential_args =[], 
                    n_samples_tune = 2000, 
                    n_samples_check = 200, 
                    n_samples_burn_in = 1000, 
                    n_samples_prod = 500, 
                    potential = potential, 
                    mass_matrix = jax.numpy.identity(3), 
                    target_AR = 0.92, 
                    stage = 3, 
                    sensibility = 0.01, 
                    step_factor=0.5, 
                    compute_freqs = True,
                    compute_hessian = False, 
                    sampler = "GHMC", 
                    RNG_key = 42,
                    initial_step_size=0.01,
                    n_steps_mode='random')



# samples_RWMH = _single_chain_RWMH(params_init,
#                                   n_samples = 100,
#                                   burn_in = 100,
#                                   step_size = 0.01,
#                                   potential = potential,
#                                   key = jax.random.PRNGKey(120))

# samples_HMC = RWMH(params_init,
#                     potential_args=(),
#                     n_samples = 100, 
#                     burn_in = 100, 
#                     step_size = 0.01, 
#                     potential = potential, 
#                     RNG_key = 120)
samples_GHMC = jnp.exp(samples_GHMC)
np.savetxt(HERE / "results_PDE_test.txt", samples_GHMC)