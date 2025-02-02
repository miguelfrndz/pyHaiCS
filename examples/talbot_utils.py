import jax
import os, sys
sys.path.append('../')
import numpy as np
from tqdm import tqdm
import jax.numpy as jnp
import matplotlib.pyplot as plt
jax.config.update("jax_enable_x64", True)
from BesselJAX import J1
import pyHaiCS as haics
import ctypes

USE_C_VERSION = True

if USE_C_VERSION:
    # Load the compiled C shared library
    lib = ctypes.CDLL('./monte_carlo_integral.so')

    # Define the C function prototype
    lib.monte_carlo_integrate.argtypes = [
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_double),
        ctypes.c_double, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_double)
    ]

    # Utility function
    int_t = ctypes.c_int
    double_t = ctypes.c_double
    ptr_t = ctypes.POINTER(double_t)
    ptr = lambda array: array.ctypes.data_as(ptr_t)

    lib.compute_integrals.argtypes = [
        ptr_t, ptr_t, ptr_t, ptr_t, ptr_t, ptr_t, ptr_t,
        int_t, int_t, int_t, int_t,
        double_t, double_t, double_t, double_t
    ]
    lib.compute_integrals.restype = None

@jax.jit
def integrand(tau, kn, t, z, omega, epsilon=1e-3):
    u = jnp.sqrt(jnp.maximum(0, tau ** 2 - z ** 2))
    mask = u < epsilon
    result = jnp.where(mask, jnp.sin(omega * (t - tau)) * kn * 1/2, jnp.sin(omega * (t - tau)) * J1(kn * u) / u)
    return result

def g_n_rect_delta(n, config):
    '''Computes the g_n of the Rect function NORMALISED so that it tends to the delta function.
    For n =/= 0 we return g_n + g_{-n} = 2 g_n
    
    Parameters
    ----------
    n : np.ndarray of ints
        The ns for which to cumpute g_n. Must be integers.

    config : TalbotConfig
        The class containing all the parameters of the simulation.
    
    Returns
    -------
    g_n : np.ndarray of floats
        The values of the g_ns relative to each n.
    '''

    original_settings = np.seterr()
    np.seterr(divide='ignore', invalid='ignore')
    result = np.where(n == 0, 1, 2*np.sin(n * np.pi * config.w) / (np.pi * n * config.w**2)) # We multiply by 2 to account at the same time for g_n and g_(-n)
    np.seterr(**original_settings)
    return result

def perform_integrals(config):
    '''This mehtod computes the integrals in (4.8) using GSL's integration method.
    In order to minimise the size of the integrals, we only compute integrals in the intervals [t_i-1, t_i].
    To recover the original interval [z_i, t_i] we only have to use Barrow's rule.

    Note that we must have integrands independent of t, so we use the identity 
    sin(tau-t) = cos(t)sin(tau) - sin(t)cos(tau) and separate each integral into two different ones.

    Credits to Jérôme Richard, who explained in his stackoverflow's answer stackoverflow.com/a/79360271/24208929 
    how to call the GSL library from Python for the efficient evaluation of integrals.

    Parameters
    ----------
    config : TalbotConfig
        The class containing all the parameters of the simulation.

    Returns
    -------
    result : np.ndarray of floats
        The values of the integrals at (n,t_i,z_i). The array has shape (N_max, N_t, N_z).
    '''

    # We store the range of n, kn and gn in arrays of length N_max
    n_values = np.linspace(0, config.N_max-1, config.N_max, dtype=int)
    k_n_values = 2 * np.pi * n_values

    # We store the range of t, z in the arrays of dimension (N_t) and (N_z)
    t_values = np.linspace(config.z_T/config.c * config.initial_t_zT, config.z_T/config.c * config.final_t_zT, config.N_t)
    z_values = np.linspace(0, config.N_z*config.delta_z, config.N_z)

    # Preallocate the result arrays (shape: len(n_values) x len(z_values))
    x_min = np.zeros((len(t_values), len(z_values)))
    x_max = np.empty((len(t_values), len(z_values)))

    # Compute the values
    t1_values = np.roll(t_values, 1)
    t1_values[0] = 0.
    x_min = np.maximum(z_values[None, :], t1_values[:, None])  # Max between z_values[j] and t_values[i-1]
    x_max = np.maximum(z_values[None, :], t_values[:, None])  # Max between z_values[j] and t_values[i]
        
    # Prealocate the result arrays
    integral = np.zeros((len(n_values), len(t_values), len(z_values)), dtype = np.float64)

    partial_integral_sin = np.zeros((len(n_values),len(t_values),len(z_values)))
    partial_integral_cos = np.zeros((len(n_values),len(t_values),len(z_values)))

    if USE_C_VERSION:
        # Call C function
        # lib.monte_carlo_integrate(
        #     n_values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        #     k_n_values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        #     t_values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        #     z_values.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        #     x_min.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        #     x_max.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        #     ctypes.c_double(config.omega),
        #     ctypes.c_int(len(n_values)), ctypes.c_int(len(t_values)), ctypes.c_int(len(z_values)),
        #     ctypes.c_int(config.mc_samples), integral.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        # )

        # result = np.cumsum(integral, axis = 1)
        lib.compute_integrals(
            ptr(partial_integral_cos), ptr(partial_integral_sin), ptr(x_min), ptr(x_max), 
            ptr(k_n_values), ptr(t_values), ptr(z_values), 
            len(n_values), len(t_values), len(z_values), 
            100_000, 1e-7, 1e-4, config.omega, 1e-6
        )

        # Initialize the result array
        resummed_integral_sin = np.empty((len(n_values), len(t_values), len(z_values)))
        resummed_integral_cos = np.empty((len(n_values), len(t_values), len(z_values)))

        # Vectorized cumulative sum along the 't' axis (axis=1) to recover the full integrals
        resummed_integral_sin[:, :, :] = np.cumsum(partial_integral_sin[:, :, :], axis=1)
        resummed_integral_cos[:, :, :] = np.cumsum(partial_integral_cos[:, :, :], axis=1)


        # Initialize the result array
        result = np.empty((len(n_values), len(t_values), len(z_values)))
        result = np.sin(config.omega * t_values[None,:,None]) * resummed_integral_cos \
            - np.cos(config.omega * t_values[None,:,None]) * resummed_integral_sin # sin(tau-t) = cos(t)sin(tau) - sin(t)cos(tau)

    else:
        # Perform the integrals
        for n in tqdm(range(len(n_values))):
            for i in range(len(t_values)):
                for j in range(len(z_values)):
                    # Monte Carlo integration
                    samples = np.random.uniform(x_min[i, j], x_max[i, j], config.mc_samples)
                    integrand_values = integrand(samples, k_n_values[n], t_values[i], z_values[j], config.omega)
                    integral[n, i, j] = (x_max[i, j] - x_min[i, j]) * np.mean(integrand_values)

        # Initialize the result array
        result = np.empty((len(n_values), len(t_values), len(z_values)))

        # Vectorized cumulative sum along the 't' axis (axis=1) to recover the full integrals
        result[:, :, :] = np.cumsum(integral[:, :, :], axis = 1)

    return result

def generate_coeffs(config):
    '''Computes the coefficients $c_n(t,z)$ for each allowed value of n,t,z for the parameters in config.
    
    Parameters
    ----------
    config : TalbotConfig
        The class containing all the parameters of the simulation.
    
    Returns
    -------
    coeffs : np.ndarray of floats
        The values of the coefficients $c_n(t,z)$. The array is of shape (N_max, N_t, N_z).
    '''

    # We store the range of t, z in the arrays of dimension (N_t) and (N_z)
    t_values = np.linspace(config.z_T/config.c * config.initial_t_zT, config.z_T/config.c * config.final_t_zT, config.N_t)
    z_values = np.linspace(0, config.N_z*config.delta_z, config.N_z)

    # We store the range of n, kn and gn in arrays of length N_max
    n_values = np.linspace(0, config.N_max-1, config.N_max, dtype=int)
    g_n_values = g_n_rect_delta(n_values, config)
    k_n_values = 2 * np.pi * n_values


    # We perform the integrals
    integrals_array = perform_integrals(config)


    # Component 1: - k_n_values[n] * integrals_array[n,t,z]
    integrals_part = - k_n_values[:,np.newaxis,np.newaxis] * z_values[np.newaxis,np.newaxis,:] * integrals_array

    # Component 2: sin(omega * (t_values[i] - z_values[j])) * Heaviside(t_values[i] - z_values[j])
    heaviside_part = np.sin(config.omega * (t_values[np.newaxis,:,np.newaxis] - z_values[np.newaxis,np.newaxis,:])) * np.heaviside(t_values[np.newaxis,:,np.newaxis] - z_values[np.newaxis,np.newaxis,:], 0.5)

    # Combine the components
    coeffs = g_n_values[:,np.newaxis,np.newaxis] * (integrals_part + heaviside_part)

    return coeffs

def generate_amplitude_field(config):
    '''Computes the value of the solution $u(t,x,z)$ for each allowed value of t, x and z for the parameters in config.
    
    Parameters
    ----------
    config : TalbotConfig
        The class containing all the parameters of the simulation.
    
    Returns
    -------
    field : np.ndarray of floats
        The values of the field $u(t,x,z)$ at each t,x,z. The array is of shape (N_t, N_x, N_z).
    '''


    # We compute the coefficients $c_n (t,z)$
    coeffs = generate_coeffs(config)

    # We compute the wave numbers
    k_n = 2 * np.pi * np.arange(config.N_max)

    # Create the grid for the cosine terms
    x_grid = np.arange(config.N_x) * config.delta_x
    cos_values = np.cos(np.outer(k_n, x_grid))
    del x_grid, k_n

    field = np.zeros([config.N_t, config.N_x, config.N_z])

    # Iterate over chunks of n axis to avoid memory problems
    chunk_size = 2  # Choose a reasonable chunk size based on available memory
    for i in tqdm(range(0, coeffs.shape[0], chunk_size)):
        chunk_coeffs = coeffs[i:i+chunk_size]
        chunk_cos_values = cos_values[i:i+chunk_size]
        
        # Compute the chunk of the contribution to E from the cosine term. 
        # For this we expand dimensions of coeffs and cos_values to match the desired broadcast shape.
        # New shapes:       (N_max, N_t, 1, N_z)        (N_max, 1, N_x,1)
        field_update_chunk = chunk_coeffs[:, :, np.newaxis, :] * chunk_cos_values[:, np.newaxis, :, np.newaxis]
        
        # Sum over n (along axis 0) and add to the field
        field += np.sum(field_update_chunk, axis=0)
        del field_update_chunk
    del cos_values, coeffs, chunk_coeffs, chunk_cos_values

    return field

def resize_field(field, config):
    """ Extends the domain of the solution from $0 \leq x \leq d/2$ to $-d \leq x \leq d$ using its symmetry.
    
    Parameters
    ----------
    field : np.ndarray of floats
        The values of the field $u(t,x,z)$ at each t,x,z for $0 \leq x \leq d/2$. The array is of shape (N_t, N_x, N_z).
    config : TalbotConfig
        The class containing all the parameters of the simulation.
    
    Returns
    -------
    resized_field : np.ndarray of floats
        The values of the field $u(t,x,z)$ at each t,x,z with for $-d \leq x \leq d$. Its shape is (N_t, 4 N_x, N_z).
    """
    # We allocate the new array
    resized_field = np.empty([config.N_t, 4 * config.N_x, config.N_z])

    # Define an auxiliary array
    reversed_order = np.array([n for n in range(config.N_x - 1, -1, -1)])

    resized_field[:, 0:config.N_x, :] = field[:,0:config.N_x,:]
    resized_field[:,config.N_x:2*config.N_x,:] = field[:,reversed_order, :]
    resized_field[:,2*config.N_x:3*config.N_x,:] = field[:,0:config.N_x,:]
    resized_field[:,3*config.N_x:4*config.N_x,:] = field[:,reversed_order, :]
    return resized_field

def plot_field(field, config, folder_path, title, file_name, save_field = False, difference = False, cmap = 'gray'):
    '''Plots the field at time t_i and saves the image in folder_path as a PNG.

    Parameters
    ----------
    field : np.ndarray of floats
        Field to be plotted. Must of size (N_t, 4 N_x, N_z).
    config : TalbotConfig
        Class storing the parameters of the simulation.
    folder_path : string
        Path where the image must be stored. The folder MUST exist.
    title : string
        Title of the plot
    file_name : string
        Name of the file.
    save_field : bool, optional
        Whether the image should be saved as a txt file alongside the PNG image.
    difference : bool, optional
        Whether the values of the image run from -A^2 to A^2 (True) or from 0 to A^2 (False).
    cmap : str, optional
        Colormap used for the images. See documentation of matplotlib. Some recommended options are gray and turbo.

    Returns
    -------
    None
    '''
    cm = 1/2.54
    plt.figure(figsize=(32*cm, 18*cm))
    plt.title(title, fontsize = 20, y = 1.05)

    # Plot the Field
    # Some nice options are gray and turbo 
    im = plt.imshow(field, cmap = cmap, vmin = 0, vmax = (config.d/config.w)**2, interpolation = 'none')

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
    if difference:
        cbar = plt.colorbar(im, ticks=[-(config.d/config.w)**2, -(config.d/config.w)**2/2, 0., (config.d/config.w)**2/2, (config.d/config.w)**2], fraction = 0.0458 * config.N_z/(4 * config.N_x), pad = 0.04, shrink = 0.9)
        cbar.set_label(label = 'Intensity of the field', fontsize = 18)
        cbar.ax.set_yticklabels(['$-A^2$', '$-\\dfrac{A^2}{2}$', '$0$', '$\\dfrac{A^2}{2}$', '$A^2$'], fontsize = 16)
    else:
        cbar = plt.colorbar(im, ticks=[0., (config.d/config.w)**2/4, (config.d/config.w)**2/2, 3*(config.d/config.w)**2/4, (config.d/config.w)**2], fraction = 0.0458 * config.N_z/(4 * config.N_x), pad = 0.04, shrink = 0.9)
        cbar.set_label(label = 'Intensity of the field', fontsize = 18)
        cbar.ax.set_yticklabels(['$0$', '$\\dfrac{A^2}{4}$', '$\\dfrac{A^2}{2}$', '$\\dfrac{3A^2}{4}$', '$A^2$'], fontsize = 16)
    #cbar.set_label(label = 'Amplitude of the field', fontsize = 18)
    #cbar.ax.set_yticklabels(['$-A$', '$-\\dfrac{A}{2}$', '$0$', '$\\dfrac{A}{2}$', '$A$'], fontsize = 16) 

    plt.savefig(os.path.join(folder_path, file_name), bbox_inches = 'tight', dpi = 300)  
    plt.close()

    if save_field:
        # Save the field at time t_i to a txt file
        np.savetxt(os.path.join(folder_path, file_name), field, delimiter=',')

def video_from_images(images_path, output_name, fps=24):
    '''Creates the video showcasing the formation of the Talbot effect.

    Parameters
    ----------
    images_path : string
        Path where the images are stored. The images name must be sorted and end with .png to be identified by this function. The video will be stored in this same path.
    output_name : string
        Name of the output file.
    fps : int, optional
        Framerate of the video

    Returns
    -------
    None
    '''

    if not os.path.exists(os.path.dirname(images_path)): # We make sure that the images_path exists.
        os.makedirs(os.path.dirname(images_path))
        
    command = f'ffmpeg -framerate {fps} -pattern_type glob -i "{images_path}/*.png" -vf "scale=3288:1928" -c:v libx264 -pix_fmt yuv420p "{output_name}"'
    os.system(command) # We create the video