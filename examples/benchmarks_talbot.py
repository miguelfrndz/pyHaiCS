import jax
import os, sys, io
import numpy as np
from datetime import datetime
from tqdm import tqdm
import jax.numpy as jnp
import matplotlib.pyplot as plt
from talbot_utils import generate_amplitude_field, resize_field, plot_field

class TalbotConfig:
    def __init__(self):
        """
        Parameters of the simulation
        -------------------------
        A : Amplitude of the source.
        c : Speed of light. Fixed at 1.
        d : Distance between the gratings. We fix it = 1.
        _lambda : Wavelength λ of the source.
        w : Width of the gratings.

        Related relevant magnitudes
        ---------------------------
        omega : Angular frequency of the source = 2∏C/λ.
        z_T : Talbot distance = λ/(1 - √(1 - λ/d^2)).

        Grid parameters
        ---------------
        N_t : Number of samples in the t direction.
        N_x : Number of samples in the x direction.
        N_z : Number of samples in the z direction.

        Simulation parameters
        ---------------------
        N_max : Value of the series' cutoff. We want N_max >> d/λ.
        initial_t_zT : Initial time / Z_t.
        final_t_zT : Final time / Z_t.

        Other relevant magnitudes
        -------------------------
        delta_t : Time between points = self.z_T/self.c/(self.N_t-1) * (self.final_t_zT - self.initial_t_zT).
        delta_x : X-Distance between points = self.d/2/self.N_x.
        delta_z : Z-Distance between points = self.z_T/self.N_z.
        """
        self.A = 1 # Amplitude of source
        self.c = 1 # Speed of light
        self.d = 1 # Distance between gratings we fix it = 1
        self._lambda = self.d / 10 # Wavelength (TODO: Change to /100)
        self.w = 2 * self._lambda # Width of the gratings

        # Other related relevant magnitudes
        self.omega = 2 * jnp.pi * self.c / self._lambda # Angular frequency of the source
        self.z_T = self._lambda/(1 - np.sqrt(1-(self._lambda/self.d) ** 2)) # Talbot distance = 2 d^2/λ

        # Grid Simulation parameters
        self.N_x = 27 * 10 # Number of samples in x direction
        self.N_z = 192 * 10 # Number of samples in z direction
        self.N_t = 250 # Number of samples in time

        # Simulation parameters
        self.N_max = int(self.d / self._lambda * 5) # Number of terms in the series
        self.initial_t_zT = 0 # Initial time / Z_t
        self.final_t_zT = 2 # Final time / Z_t

        # Other relevant magnitudes
        self.delta_t = self.z_T/self.c/(self.N_t - 1) * (self.final_t_zT - self.initial_t_zT) # Time between photos
        self.delta_x = self.d/2/self.N_x # X-Distance between points
        self.delta_z = self.z_T/self.N_z # Z-Distance between points

        # Parameters for the Monte-Carlo simulation
        self.mc_samples = 5_000 # Number of samples for the Monte-Carlo simulation

    def __str__(self):
        params = {
            "Amplitude of the source (A)": self.A,
            "Speed of light (c)": self.c,
            "Distance between gratings (d)": self.d,
            "Wavelength (lambda)": self._lambda,
            "Width of the gratings (w)": self.w,
            "Angular frequency of the source": self.omega,
            "Talbot distance (z_T)": self.z_T,
            "Number of samples in x direction (N_x)": self.N_x,
            "Number of samples in z direction (N_z)": self.N_z,
            "Number of samples in time (N_t)": self.N_t,
            "Number of terms in the series (N_max)": self.N_max,
            "Time between photos (delta_t)": self.delta_t,
            "X-Distance between points (delta_x)": self.delta_x,
            "Z-Distance between points (delta_z)": self.delta_z,
            "Initial time / z_T": self.initial_t_zT,
            "Final time / z_T": self.final_t_zT,
            "Number of Monte-Carlo samples": self.mc_samples
        }
        output = io.StringIO()
        # Print to the string stream instead of the console
        output.write("{:<45} {:<40}\n".format('Parameter', 'Value'))
        output.write("-" * 65 + "\n")
        for key, value in params.items():
            output.write("{:<45} {:<40}\n".format(key, value))        
        result = output.getvalue()
        output.close()
        return result

if __name__ == "__main__":
    config = TalbotConfig()
    print(config)

    # Photo destination
    my_path = os.path.dirname(os.path.abspath(__file__))
    results_path = os.path.join(my_path, 'talbot_results')

    # Create the folder if it doesn't exist
    if not os.path.isdir(results_path):
        os.makedirs(results_path)

    # Create the folder for the simulation
    folder_name = 'd_λ=' + str(config.d/config._lambda) + '_w_λ=' + str(config.w/config._lambda) + '_' + datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    folder_path = os.path.join(results_path, folder_name)
    if not os.path.isdir(folder_path): # Create the folder if it doesn't exist
        os.makedirs(folder_path)

    # Save the parameters of the simulation into a file
    with open(os.path.join(folder_path, 'parameters.txt'), 'w') as file:
        file.write(str(config)) 

    # We compute the intensity of the light and extend it to -d <= x <= d
    field = generate_amplitude_field(config) # We compute the amplitude of the solution
    field = field**2 # We compute the intensity of the light
    field = resize_field(field, config) # We use the symmetry of the solution to extend the domain to -d <= x <= d

    # We create the caché folder if it doesn't exist
    cache_path = os.path.join(folder_path, 'cache')
    if not os.path.isdir(cache_path):
        os.makedirs(cache_path)

    # We plot the solution at each time t_i at cache
    for t_i in tqdm(range(0, config.N_t)):
        title = 'Intensity of the field at $t = ' + str(round(t_i * config.delta_t/(config.z_T),4)) + '\\, Z_T/c$ for $\\frac{d}{\\lambda}='+str(1/config._lambda)+'$ and $\\frac{w}{\\lambda}=' + str(config.w/config._lambda)+'$'
        file_name = 'd_λ=' + str(1/config._lambda) + '_w_λ=' + str(config.w/config._lambda)+'_' + str(t_i).rjust(len(str(config.N_t)),'0') + '_carpet.png'
        plot_field(field[t_i], config, cache_path, title, file_name, save_field = False, cmap = 'gray')

    # We plot the final image also somewhere else to store it
    final_field = field[config.N_t - 1]
    del field
    file_name = 'd_λ=' + str(1/config._lambda) + '_w_λ=' + str(config.w/config._lambda)+'_TRANSIENT_carpet.png'
    plot_field(final_field, config, folder_path, title, file_name, save_field = False, cmap = 'gray')