"""
Talbot Benchmark
"""
import sys, os
sys.path.append('../')
import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import pandas as pd
import pyHaiCS as haics
from tqdm import tqdm
import matplotlib.pyplot as plt
from BesselJAX import J0
from quadax import quadgk

class TalbotConfig:
    """
    Configuration for the Talbot benchmark.
    """
    def __init__(self) -> None:
        # Physical parameters. Might be intersting to repeat for different values.
        self.A = 1 # Amplitude of signal
        self.c = 1 # Speed of light
        self.d = 1 # Distance between gratings we fix it = 1
        self._lambda = self.d/10 # Wavelength
        self.w = 2 * self._lambda # Width of the gratings
        # Other relevant magnitudes
        self.omega = 2*jnp.pi*self.c/self._lambda # Frequency of the signal
        self.z_T = self._lambda/(1 - jnp.sqrt(1-(self._lambda/self.d)**2)) # Talbot distance = 2 d^2/λ
        # Simulation parameters
        self.N_x = 27 # Number of samples in x direction
        self.N_z = 192 # Number of samples in z direction
        self.N_t = 40 # Number of samples in time
        self.N_max = int(self.d/self._lambda * 10) # Number of terms in the series
        # Other relevant magnitudes
        self.delta_t = self.z_T/self.c/self.N_t # Time between photos
        self.delta_x = self.d/2./self.N_x # X-Distance between points
        self.delta_z = self.z_T/self.N_z # Z-Distance between points

talbot_config = TalbotConfig()
key = jax.random.PRNGKey(42)

def resize_field_unJAXed(field):
    resized_field = jnp.empty([talbot_config.N_t,4*talbot_config.N_x, talbot_config.N_z])
    reversed_order = jnp.array([n for n in range(talbot_config.N_x-1,-1,-1)])
    resized_field = resized_field.at[:, 0:talbot_config.N_x, :].set(field[:,0:talbot_config.N_x,:])
    resized_field = resized_field.at[:,talbot_config.N_x:2*talbot_config.N_x,:].set(field[:,reversed_order, :])
    resized_field = resized_field.at[:,2*talbot_config.N_x:3*talbot_config.N_x,:].set(field[:,0:talbot_config.N_x,:])
    resized_field = resized_field.at[:,3*talbot_config.N_x:4*talbot_config.N_x,:].set(field[:,reversed_order, :])
    return resized_field

@jax.jit
def obtain_c_n():
    # We create an array to store the coordinates of the grid. array_coeffs[0,1,2][n][z][t] are n,z,t, respectively. 
    coords = jnp.indices((talbot_config.N_max, talbot_config.N_t, talbot_config.N_z))
    # We normalise the array elements
    coords = coords.astype(jnp.float64)
    coords = coords.at[1,:,:,:].multiply(talbot_config.delta_t)
    coords = coords.at[2,:,:,:].multiply(talbot_config.delta_z)
    # We reshape it so it is 2-dimensional now
    coords = coords.reshape((3,-1)).T
    # We compute the coefficient for each n, t, z
    coeffs = jax.vmap(coefficient_ntz)(*coords.T)
    # We reshape coeffs back to the original grid shape
    coeffs = coeffs.reshape((talbot_config.N_max, talbot_config.N_t, talbot_config.N_z))
    return coeffs

@jax.jit
def h_n_vectorised(n):
    return jax.lax.cond(
        n == 0,   # Condition to check
        lambda _: 2 * talbot_config.w / talbot_config.d,  # If True, return 2 * w / d
        lambda _: 2 * jnp.sin(n * jnp.pi * talbot_config.w / talbot_config.d) / (jnp.pi * n),  # If False, return 2 * jnp.sin(n * jnp.pi * w / d) / (jnp.pi * n)
        operand=None  # You can pass `n` if needed for other purposes
    )

@jax.jit
def fn(x, t, z, k_n):
    return J0(k_n * jnp.sqrt((talbot_config.c * jnp.tan(x)) ** 2 - z ** 2)) * jnp.sin(talbot_config.omega * (jnp.tan(x) + t))

@jax.jit
def potential_fn(z, k_n, x):
    return -jnp.log(J0(k_n * jnp.sqrt((talbot_config.c * jnp.tan(x)) ** 2 - z ** 2)))

@jax.jit
def integrate(t,z,k_n):
    # Integration limits (with change of variable x = tan(tau), dx = 1/cos(tau)^2 dtau)
    int_lower, int_upper = jnp.arctan(jnp.abs(z)/talbot_config.c), jnp.pi/2
    # Method 1 - Numerical Integration Using quadgk (Gauss-Kronrod quadrature)
    # integral = quadgk(lambda x: fn(x, t, z, k_n) / jnp.cos(x)**2, [int_lower, int_upper])[0]
    # Method 2 - Using Hamiltonian Monte-Carlo
    x = jnp.linspace(int_lower, int_upper, 1000)
    samples_HMC = haics.samplers.hamiltonian.HMC(x, 
                        potential_args = (z, k_n),                                           
                        n_samples = 5000, burn_in = 5000, 
                        step_size = 1e-3, n_steps = 100, 
                        potential = potential_fn,  
                        mass_matrix = jnp.eye(x.shape[0]), 
                        integrator = haics.integrators.VerletIntegrator(), 
                        RNG_key = key, n_chains = 4)
    samples_HMC = jnp.mean(samples_HMC, axis = 0) # Average across chains
    integral = jnp.mean(jnp.sin(talbot_config.omega * (jnp.tan(samples_HMC) + t))/jnp.cos(samples_HMC)**2)
    return integral

@jax.jit
def coefficient_ntz(n,t,z):
    k_n = 2*jnp.pi * n/talbot_config.d

    c_n = k_n * integrate(t,z,k_n)

    c_n += jnp.sin(talbot_config.omega*(t+z/talbot_config.c))*(1-jnp.sin(k_n*(talbot_config.c*t+z)))*jnp.heaviside(t+z/talbot_config.c,0.5)
    c_n += jnp.sin(talbot_config.omega*(t-z/talbot_config.c))*(1-jnp.sin(k_n*(talbot_config.c*t-z)))*jnp.heaviside(t-z/talbot_config.c,0.5)

    c_n *= talbot_config.A * h_n_vectorised(n)/2

    return c_n

@jax.jit
def obtain_AField():
    coeffs = obtain_c_n()

    # Generate a null array of shape (N_t, N_x, N_z)
    A_field = jnp.zeros([talbot_config.N_t, talbot_config.N_x, talbot_config.N_z])

    # Generate the k_n values
    k_n = 2 * jnp.pi * jnp.arange(talbot_config.N_max) / talbot_config.d   # k_n is now a vector (N_max)

    # Create the grid for the cosine terms (broadcasting will be used)
    x_grid = jnp.arange(talbot_config.N_x) * talbot_config.delta_x  # Shape (N_x)
    cos_values = jnp.cos(jnp.outer(k_n, x_grid))  # Shape (N_max, N_x), broadcasting k_n over x_grid
    del x_grid

    # Compute the contribution to E from the cosine term. 
    # For this we expand dimensions of coeffs and cos_values to match the desired broadcast shape.
    # New shapes:       (N_max, N_t, 1, N_z)        (N_max, 1, N_x,1)
    A_update = coeffs[:, :, jnp.newaxis, :] * cos_values[:, jnp.newaxis, :, jnp.newaxis]  # Shape: (N_max, N_t, N_x, N_z)

    # Sum over n (along axis 0) to update E
    A_field += jnp.sum(A_update, axis=0)  # Shape: (N_t, N_x, N_z)
    return A_field

def plot_figure(t_i, field):

    cm = 1/2.54
    plt.figure(figsize=(32*cm, 18*cm))
    plt.title('Gauge Field at $t = ' + str(round(t_i*talbot_config.delta_t/(talbot_config.z_T/talbot_config.c),4)) + '\\, Z_T/c$ for $\\frac{d}{\\lambda}='+str(talbot_config.d/talbot_config._lambda)+'$ and $\\frac{w}{\\lambda}='+str(talbot_config.w/talbot_config._lambda)+'$', fontsize = 20, y = 1.05)

    im = plt.imshow(field[t_i], cmap='RdBu_r', vmin=-talbot_config.A, vmax=talbot_config.A, interpolation='none')

    # We label the X axis
    plt.ylabel('Grating', fontsize = 18)
    plt.ylim(0, 4*talbot_config.N_x)
    ticks_x = [0, talbot_config.N_x-1, 2*talbot_config.N_x-1, 3*talbot_config.N_x-1, 4*talbot_config.N_x-1]
    labels_x = ['$-d$', '$-\\dfrac{d}{2}$', '$0$', '$\\dfrac{d}{2}$', '$d$']
    plt.yticks(ticks_x, labels_x, fontsize = 16)

    # We label the Z axis
    plt.xlabel('Propagation of light ---->', fontsize = 18)
    plt.xlim(0, talbot_config.N_z-1)
    ticks_z = [0, talbot_config.N_z/4, talbot_config.N_z/2, 3*talbot_config.N_z/4, talbot_config.N_z]
    labels_z = ['$0$', '$\\dfrac{1}{4} Z_T$', '$\\dfrac{1}{2} Z_T$', '$\\dfrac{3}{4} Z_T$', '$Z_T$']
    plt.xticks(ticks_z, labels_z, fontsize = 16)

    # We make the colorbar look nice
    # cbar = plt.colorbar(im, ticks=[0, A**2/4, A**2/2, 3*A**2/4, A**2], fraction=0.0458*N_z/(4*N_x), pad=0.04, shrink=0.9)  # Muestra la barra de color
    # cbar.set_label(label = 'Intensity of the field', fontsize = 18)
    # cbar.ax.set_yticklabels(['$0$', '$\\dfrac{A^2}{4}$', '$\\dfrac{A^2}{2}$', '$\\dfrac{3A^2}{4}$', '$A^2$'], fontsize = 16)
    
    cbar = plt.colorbar(im, ticks=[-talbot_config.A, -talbot_config.A/2, 0., talbot_config.A/2, talbot_config.A], fraction=0.0458*talbot_config.N_z/(4*talbot_config.N_x), pad=0.04, shrink=0.9)  # Muestra la barra de color
    cbar.set_label(label = 'Amplitude of the field', fontsize = 18)
    cbar.ax.set_yticklabels(['$-A$', '$-\\dfrac{A}{2}$', '$0$', '$\\dfrac{A}{2}$', '$A$'], fontsize = 16)

    file_name = 'd_λ='+str(talbot_config.d/talbot_config._lambda)+'_w_λ='+str(talbot_config.w/talbot_config._lambda)+'_' + str(t_i) + '_carpet.pdf'
    my_path = os.path.dirname(os.path.abspath(__file__))
    folder_name = 'd_λ='+str(talbot_config.d/talbot_config._lambda)+'_w_λ='+str(talbot_config.w/talbot_config._lambda)
    folder_path = os.path.join(my_path, folder_name)
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)
    plt.savefig(os.path.join(folder_path, file_name), bbox_inches='tight')  
    plt.close()
    return

if __name__ == "__main__":
    print(f"Running pyHaiCS v.{haics.__version__}")
    A_field = obtain_AField()
    A_field_resized = resize_field_unJAXed(A_field)
    for t_i in tqdm(range(0,talbot_config.N_t)):
        plot_figure(t_i,A_field_resized)