import jax
import os, sys
import numpy as np
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
        self._lambda = self.d / 10 # Wavelength
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
        }
        
        print("{:<45} {:<40}".format('\nParameter', 'Value'))
        print("-" * 65)
        for key, value in params.items():
            print("{:<45} {:<40}".format(key, value))
        return ""

if __name__ == "__main__":
    config = TalbotConfig()
    print(config)

    # Photo destination
    my_path = os.path.dirname(os.path.abspath(__file__))
    folder_name = 'd_λ=' + str(config.d/config._lambda) + '_w_λ=' + str(config.w/config._lambda)
    folder_path = os.path.join(my_path, folder_name)

    # Create the folder if it doesn't exist
    if not os.path.isdir(folder_path):
        os.makedirs(folder_path)

    field = generate_amplitude_field(config)
    field = resize_field(field, config)
    for t_i in tqdm(range(0, config.N_t)):
        plot_field(t_i, field, config, folder_path, save_field = False)