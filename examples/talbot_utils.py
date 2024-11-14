import jax
import os, sys
import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
from functools import partial
from quadax import quadgk
jax.config.update("jax_enable_x64", True)

def integrate(t, z, k_n, config):
    @jax.jit
    def func(x):
        return jnp.sin(config.omega * (t - x / config.c)) * jnp.sin(k_n * x)
    return quadgk(lambda x: func(x), [0, jnp.pi/2])[0]

def h_n_vectorised(n, config):
    return jax.lax.cond(n == 0, lambda _: 2 * config.w / config.d,
            lambda _: 2 * jnp.sin(n * jnp.pi * config.w / config.d) / (jnp.pi * n),
            operand = None
        )

def coefficient_ntz(n, t, z, config):
    k_n = 2 * jnp.pi * n / config.d
    c_n = k_n * integrate(t, z, k_n, config)
    c_n += jnp.sin(config.omega * (t + z / config.c))*(1 - jnp.sin(k_n * (config.c * t + z))) * jnp.heaviside(t + z / config.c , 0.5)
    c_n += jnp.sin(config.omega * (t - z / config.c))*(1 - jnp.sin(k_n * (config.c * t - z))) * jnp.heaviside(t - z / config.c , 0.5)
    c_n *= config.A * h_n_vectorised(n, config) / 2
    return c_n

def generate_coeffs(config):
    # We create an array to store the coordinates of the grid
    coords = jnp.indices((config.N_max, config.N_t, config.N_z))
    
    # We normalise the array elements
    coords = coords.astype(jnp.float64)
    coords = coords.at[1,:,:,:].multiply(config.delta_t)
    coords = coords.at[2,:,:,:].multiply(config.delta_z)

    # We reshape it so it is 2-dimensional now
    coords = coords.reshape((3,-1)).T
    
    # We compute the coefficient for each n, t, z
    coeffs = jax.vmap(partial(coefficient_ntz, config = config))(*coords.T)

    # We reshape coeffs back to the original grid shape
    coeffs = coeffs.reshape((config.N_max, config.N_t, config.N_z))

    return coeffs

def generate_amplitude_field(config):
    coeffs = generate_coeffs(config)
    field = jnp.zeros([config.N_t, config.N_x, config.N_z])
    k_n = 2 * jnp.pi * jnp.arange(config.N_max) / config.d
    # Create the grid for the cosine terms
    x_grid = jnp.arange(config.N_x) * config.delta_x
    cos_values = jnp.cos(jnp.outer(k_n, x_grid))
    del x_grid
    # Compute the contribution to E from the cosine term. 
    # For this we expand dimensions of coeffs and cos_values to match the desired broadcast shape.
    # New shapes:       (N_max, N_t, 1, N_z)        (N_max, 1, N_x,1)
    field_update = coeffs[:, :, jnp.newaxis, :] * cos_values[:, jnp.newaxis, :, jnp.newaxis]
    # Sum over n (along axis 0) to update E
    field += jnp.sum(field_update, axis = 0)  # Shape: (N_t, N_x, N_z)
    return field

def resize_field(field, config):
    resized_field = np.empty([config.N_t, 4 * config.N_x, config.N_z])
    reversed_order = np.array([n for n in range(config.N_x - 1, -1, -1)])
    resized_field[:, 0:config.N_x, :] = field[:,0:config.N_x,:]
    resized_field[:,config.N_x:2*config.N_x,:] = field[:,reversed_order, :]
    resized_field[:,2*config.N_x:3*config.N_x,:] = field[:,0:config.N_x,:]
    resized_field[:,3*config.N_x:4*config.N_x,:] = field[:,reversed_order, :]
    return resized_field

def plot_field(t_i, field, config, folder_path):
    cm = 1/2.54
    plt.figure(figsize=(32*cm, 18*cm))
    plt.title('Gauge Field at $t = ' + str(round(t_i * config.delta_t/(config.z_T/config.c),4)) + '\\, Z_T/c$ for $\\frac{d}{\\lambda}='+str(config.d/config._lambda)+'$ and $\\frac{w}{\\lambda}=' + str(config.w/config._lambda)+'$', fontsize = 20, y = 1.05)
    # Plot the Gauge Field
    im = plt.imshow(field[t_i], cmap='RdBu_r', vmin=-config.A, vmax=config.A, interpolation='none')
    # Label the X axis and set the ticks
    plt.ylabel('Grating', fontsize = 18)
    plt.ylim(0, 4 * config.N_x)
    ticks_x = [0, config.N_x-1, 2 * config.N_x - 1, 3 * config.N_x - 1, 4 * config.N_x - 1]
    labels_x = ['$-d$', '$-\\dfrac{d}{2}$', '$0$', '$\\dfrac{d}{2}$', '$d$']
    plt.yticks(ticks_x, labels_x, fontsize = 16)
    # Label the Y axis and set the ticks
    plt.xlabel('Propagation of light ---->', fontsize = 18)
    plt.xlim(0, config.N_z - 1)
    ticks_z = [0, config.N_z/4, config.N_z/2, 3 * config.N_z/4, config.N_z]
    labels_z = ['$0$', '$\\dfrac{1}{4} Z_T$', '$\\dfrac{1}{2} Z_T$', '$\\dfrac{3}{4} Z_T$', '$Z_T$']
    plt.xticks(ticks_z, labels_z, fontsize = 16)
    # Add the colorbar
    cbar = plt.colorbar(im, ticks=[-config.A, -config.A/2, 0., config.A/2, config.A], fraction = 0.0458 * config.N_z/(4 * config.N_x), pad = 0.04, shrink = 0.9)
    cbar.set_label(label = 'Amplitude of the field', fontsize = 18)
    cbar.ax.set_yticklabels(['$-A$', '$-\\dfrac{A}{2}$', '$0$', '$\\dfrac{A}{2}$', '$A$'], fontsize = 16)

    file_name = 'd_λ=' + str(config.d/config._lambda) + '_w_λ=' + str(config.w/config._lambda)+'_' + str(t_i) + '_carpet.pdf'
    plt.savefig(os.path.join(folder_path, file_name), bbox_inches='tight')  
    plt.close()