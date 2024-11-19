"""
Example of a SEIR (Susceptible-Exposed-Infectious-Remove) compartmental mechanistic 
(i.e., the disease dynamics are purely governed by differential equations) dynamic 
epidemiological model (w/ a time-dependent transmission rate parametrized 
using Bayesian P-splines) applied to modeling the COVID-19 incidence data in the 
Basque Country (Spain).

Originally published in:
"Dynamic SIR/SEIR-like models comprising a time-dependent transmission rate: Hamiltonian Monte Carlo approach with applications to COVID-19"
by Hristo Inouzhe, María Xosé Rodríguez-Álvarez, Lorenzo Nagar, Elena Akhmatskaya
"""

import sys, os
sys.path.append('../')

import jax
import numpy as np
import pyHaiCS as haics
import jax.numpy as jnp
import matplotlib.pyplot as plt
from scipy.stats import truncnorm
from scipy.interpolate import BSpline
from scipy.integrate import solve_ivp
from functools import partial

DEBUG = False

def log_prior(coeffs):
    """
    Log-prior of the parameters of the SEIR model.
    """
    alpha, gamma, E0, phi, beta = coeffs[0], coeffs[1], coeffs[2], coeffs[3], coeffs[4:]
    alpha_prior = jax.scipy.stats.norm.logpdf(alpha, 0.5, 0.05)
    gamma_prior = jax.scipy.stats.truncnorm.logpdf(gamma, (gamma_lower_trunc_bound - gamma_mu) / gamma_sigma, 
                  (gamma_upper_trunc_bound - gamma_mu) / gamma_sigma, 
                  loc = gamma_mu, scale = gamma_sigma)
    E0_prior = jax.scipy.stats.norm.logpdf(E0, 21.88, 7.29)
    phi_prior = jax.scipy.stats.expon.logpdf(1/phi, 10)
    beta_prior = jax.scipy.stats.norm.logpdf(beta, -1.6, 0.5).sum()
    if DEBUG: print(f"Alpha prior: {alpha_prior}, Gamma prior: {gamma_prior}, E0 prior: {E0_prior}, Phi prior: {phi_prior}, Beta prior: {beta_prior}")
    return alpha_prior + gamma_prior + E0_prior + phi_prior + beta_prior

def log_likelihood(data, state, params):
    """
    Log-likelihood of the SEIR model.
    """
    alpha, gamma, E0, phi, beta = params[0], params[1], params[2], params[3], params[4:]
    # Solve the ODE system using the initial parameters
    t_span = (0, len(data))
    solution = solve_ivp(SEMIKR_ODE, t_span, state, args=(params, M, K, spline_basis), t_eval=jnp.arange(0, len(data), 1))
    # Extract the solution of the ODE system
    S, E, I, R, C = solution.y[0], solution.y[1:M + 1], solution.y[M + 1:M + K + 1], solution.y[M + K + 1], solution.y[M + K + 2]
    # C contains the cumulative incidence data, so we need to compute the daily incidence
    C_pred = jnp.zeros(len(C))
    C_pred = C_pred.at[0].set(C[0])
    C_pred = C_pred.at[1:].set(C[1:] - C[:-1])
    # Now extract samples from a negative binomial distribution with n = predicted_daily_incidence, p = phi
    return jax.scipy.stats.nbinom.logpmf(data, C_pred, phi).sum()

def potential_fn(data, state, params):
    """
    Potential function of the SEIR model.
    """
    if DEBUG: print(f"Log-prior: {log_prior(params)}, Log-likelihood: {log_likelihood(data, params)}")
    return -log_prior(params) - log_likelihood(data, state, params)

def potential_fn_grad(data, state, params):
    """
    Gradient of the potential function of the SEIR model.
    """
    # Print shapes of everything
    print(f"Data shape: {data.shape}, State shape: {state.shape}, Params shape: {params.shape}")
    grad_prior = jax.grad(log_prior)(params)
    grad_likelihood = jax.grad(log_likelihood)(data, state, params)
    print(f"Grad prior: {grad_prior}, Grad likelihood: {grad_likelihood}")
    return -grad_prior - grad_likelihood

def SEMIKR_ODE(t, state, params, M, K, beta_t):
    """
    System of ODEs of the SEMIKR model.
    """
    # Unpack the state of the system (S, E1, ..., EM, I1,..., IK, R, C)
    S = state[0]
    E = state[1:M + 1]
    I = state[M + 1:M + K + 1]
    R = state[M + K + 1]
    C = state[M + K + 2]
    alpha, gamma = params[0], params[1]
    # Compute the time-dependent transmission rate
    beta = beta_t(t)
    # Compute the derivatives of the system
    dSdt = -beta * S * I.sum() / population_size
    dEdt = np.zeros(M)
    dEdt[0] = beta * S * I.sum() / population_size - M * alpha * E[0]
    for m in range(1, M):
        dEdt[m] = M * alpha * E[m - 1] - M * alpha * E[m]
    dIdt = np.zeros(K)
    dIdt[0] = M * alpha * E[M - 1] - K * gamma * I[0]
    for k in range(1, K):
        dIdt[k] = K * gamma * I[k - 1] - K * gamma * I[k]
    dRdt = K * gamma * I.sum()
    dCdt = beta * S * I.sum() / population_size
    return np.concatenate([[dSdt], dEdt, dIdt, [dRdt, dCdt]])

def corrected_incidence_data(original_data):
    """
    Correct the incidence data for underreporting.
    """
    corrected_data = np.copy(original_data)
    corrected_data[:92] /= 0.15
    corrected_data[93:281] /= (0.15 + (0.54 - 0.15)*(np.arange(93, 281) - 92)/(231 - 92))
    corrected_data[281:] /= 0.54
    return corrected_data

def plot_incidence_curve(original_data, corrected_data, dates, save = False):
    plt.figure(figsize = (10, 6))
    plt.plot(dates, original_data, 'o', label = 'Daily Incidence Data', color = 'black', markersize = 3, alpha = 0.25)
    plt.plot(dates, corrected_data, 'o', label = 'Incidence Data (Corrected for Undereporting)', color = 'black', markersize = 3)
    # plt.plot(dates, model_pred, label = 'Model Prediction', color = 'blue')
    plt.legend()
    plt.title("Daily COVID-19 Incidence in the Basque Country")
    plt.xlabel('Date')
    plt.ylabel('Daily Incidence')
    plt.legend()
    plt.grid()
    if save: plt.savefig('Daily_Incidence_Curve_Basque.pdf')
    plt.show()

filePath = filePath = os.path.join(os.path.dirname(__file__), f"../pyHaiCS/benchmarks/Epidemiological/Basque_Country_covid19_SIR_data.txt")
original_data = np.loadtxt(filePath)[:-1] # Last instance is the total population
population_size = int(np.loadtxt(filePath)[-1])
corrected_data = corrected_incidence_data(original_data)
# Initial date: 10th of February 2020. Last date: 31st of January 2021
initial_date, last_date = np.datetime64('2020-02-10'), np.datetime64('2021-01-31')
dates = np.arange(initial_date, last_date + 1, dtype='datetime64[D]')

# Learning rate for gradient descent optimization
learning_rate, grad_steps = 1e-7, 3000

# Initialize the parameters of the SEIR Model
alpha = np.random.normal(0.5, 0.05) # Inverse of the average time being exposed
gamma_lower_trunc_bound, gamma_upper_trunc_bound = 1/30, 1
gamma_mu, gamma_sigma = 0.1, 0.015
# Inverse of the average time being infectious
gamma = truncnorm.rvs((gamma_lower_trunc_bound - gamma_mu) / gamma_sigma, 
                  (gamma_upper_trunc_bound - gamma_mu) / gamma_sigma, 
                  loc = gamma_mu, scale = gamma_sigma)
E0 = np.random.normal(21.88, 7.29) # Initial number of exposed individuals
# phi = np.random.exponential(10) # Dispersion parameter
phi = 0.005 # Dispersion parameter
beta = jnp.array([-1.6 for _ in range(15)]) # Spline coefficients
# Initial params of the SEIR model: alpha, gamma, E0, phi, beta
params = jnp.array([alpha, gamma, E0, phi, *beta])

# Initial state of the SEIR model: S, E1, ..., EM, I1,..., IK, R, C
M, K = 1, 3
E_init = jnp.array([E0] + [0 for _ in range(M - 1)])
I_init = jnp.array([0 for _ in range(K)])
state = jnp.concatenate([jnp.array([population_size - E0]), E_init, I_init, jnp.array([0, E0])])
# Define the spline basis functions for the time-dependent transmission rate (over time)
knots = jnp.linspace(0, 1, 12)
spline_basis = lambda t: jnp.exp(BSpline(knots, beta, k = 3)(t))

# Step 1: Optimize the parameters of the SEIR model using gradient descent
for i in range(grad_steps):
    params = params - learning_rate * potential_fn_grad(corrected_data, state, params)
    print(f"Step {i}, Params: {params}")
sys.exit(0)

params_samples_HMC = haics.samplers.hamiltonian.HMC(params, 
                            potential_args = (corrected_data, ),                                           
                            n_samples = 1000, burn_in = 100, 
                            step_size = 1e-3, n_steps = 100, 
                            potential = potential_fn,  
                            mass_matrix = jnp.eye(params.shape[0]), 
                            integrator = haics.integrators.VerletIntegrator(), 
                            RNG_key = 120, n_chains = 4)

print("Results for HMC")
haics.utils.metrics.compute_metrics(params_samples_HMC, thres_estimator = 'var_trunc', normalize_ESS = True)

# Average across chains
params_samples_HMC = jnp.mean(params_samples_HMC, axis = 0)

plot_incidence_curve(original_data, corrected_data, dates)