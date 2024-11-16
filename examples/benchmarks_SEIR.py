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
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import truncnorm

def plot_incidence_curve(data, dates):
    plt.figure(figsize = (10, 6))
    plt.plot(dates, data, 'o', label = 'Daily Incidence Data', color = 'blue', markersize = 3)
    plt.title("Daily COVID-19 Incidence in the Basque Country")
    plt.xlabel('Date')
    plt.ylabel('Daily Incidence')
    # plt.legend()
    plt.grid()
    plt.show()

filePath = filePath = os.path.join(os.path.dirname(__file__), f"../pyHaiCS/benchmarks/Epidemiological/Basque_Country_covid19_SIR_data.txt")
data = np.loadtxt(filePath)[:-1] # Last instance is the total population
# Initial date: 10th of February 2020. Last date: 31st of January 2021
initial_date, last_date = np.datetime64('2020-02-10'), np.datetime64('2021-01-31')
dates = np.arange(initial_date, last_date + 1, dtype='datetime64[D]')
plot_incidence_curve(data, dates)

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
beta = np.array([-1.6 for _ in range(15)]) # Spline coefficients

coeffs = np.array([alpha, gamma, E0, phi, *beta])

