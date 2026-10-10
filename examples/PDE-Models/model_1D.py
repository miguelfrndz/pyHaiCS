##~~~~~Functions for 1D solver~~~~~~
##Inputs:
# 
#   u0 = Initial state
#   L = length of one half of the domain (domain goes from -L to L)
#   N = number of cells
#   dt = time step 
#   nsteps = maximum number of steps of the simulation
#   All other arguments after that are the specific parameters of the potentials

import numpy as np
import jax
import jax.numpy as jnp
import equinox as eqx
from scipy.integrate import ode 
import scipy.signal
import diffrax

jax.config.update("jax_enable_x64", True)

class Model1D(eqx.Module):
    """
    A JAX-based solver for a 1D reaction-diffusion-advection equation.

    This class is an Equinox Module, which separates static parameters from
    JAX-traceable arrays, making it compatible with JIT compilation.
    """
    # Parameters of the model (static)
    r: float = eqx.field(static = True)
    K: float = eqx.field(static = True)
    aw: float = eqx.field(static = True)
    sigma: float = eqx.field(static = True)

    # Parameters of the model (traced)
    D: float
    m: float

    # Discretization parameters (static)
    L: float = eqx.field(static = True)
    N: int = eqx.field(static = True)
    dx: float = eqx.field(static = True)
    dt: float = eqx.field(static = True)
    nsteps: int = eqx.field(static = True)

    # Grid arrays
    x: jax.Array
    xedges: jax.Array
    
    def __init__(self, u0_shape, L, N, dt, nsteps, D, r, K, m, sigma, aw):
        """Initializes the model and grid."""
        self.D = D
        self.r = r
        self.K = K
        self.m = m
        self.aw = aw
        self.sigma = sigma
        self.L = L
        self.N = N
        self.dx = 2 * L / N
        self.dt = dt
        self.nsteps = nsteps
        
        self.x = jnp.linspace(-L + self.dx / 2, L - self.dx / 2, N)
        self.xedges = jnp.linspace(-L, L, N + 1)
    
    # def dudt(self, t, u, args):
    #     """
    #     Computes the time derivative of the state u, i.e., du/dt.
    #     This function has the standard signature for diffrax solvers: f(t, y, args).
    #     """
    #     if self.m == 1:
    #         diffusion_term = self.D * self._discrete_laplacian_1d(u)
    #         change = diffusion_term + self._growth_term(u)
    #     else:
    #         flux = self._flux(u)
    #         advection_term = -(flux[1:] - flux[:-1]) / self.dx
    #         change = advection_term + self._growth_term(u)
    #     return change
    
    def dudt(self, t, u, args):
        """
        Computes the time derivative of the state u, i.e., du/dt.
        This function has the standard signature for diffrax solvers: f(t, y, args).
        """
        u = jnp.clip(u, 1e-12, None)
        
        def linear_diffusion(operand):
            diffusion_term = self.D * self._discrete_laplacian_1d(u)
            return diffusion_term + self._growth_term(u)

        def nonlinear_diffusion(operand):
            flux = self._flux(u)
            advection_term = -(flux[1:] - flux[:-1]) / self.dx
            return advection_term + self._growth_term(u)

        return jax.lax.cond(
            self.m == 1.0,
            linear_diffusion,
            nonlinear_diffusion,
            operand=0.0
        )
    
    @eqx.filter_jit
    def solve(self, u0):
        """
        Solves the ODE problem using diffrax.
        """
        # Set up the ODE term
        term = diffrax.ODETerm(self.dudt)

        # Set up the solver
        solver = diffrax.Dopri5() # Equivalent to 'dopri5' in SciPy
       
        # Define the time interval and save points
        t_span = (0.0, self.nsteps * self.dt)
        saveat = diffrax.SaveAt(ts=jnp.linspace(t_span[0], t_span[1], self.nsteps + 1))

        # Use a stepsize controller for adaptive timestepping if needed
        # For a fixed step, we can set dt0=self.dt and remove the controller
        stepsize_controller = diffrax.PIDController(rtol=1e-5, atol=1e-5)
        
        # Solve the differential equation
        sol = diffrax.diffeqsolve(
            term,
            solver,
            t0=t_span[0],
            t1=t_span[1],
            dt0=self.dt,
            y0=u0,
            saveat=saveat, 
            throw=True,
            stepsize_controller=stepsize_controller,
            max_steps = 1500,
            args=self # Pass self as args to dudt
        )
        return sol.ts, sol.ys
    
    # def solveODE(self):
        
    #     ODE = ode(self.dudt).set_integrator('dopri5', rtol=1e-5)
    #     ODE.set_initial_value(self.u0, 0)
        
    #     t = 0
    #     count = 0 
        
    #     u_sol = []
    #     u_sol.append(self.u0)
        
    #     dt = self.dt
        
     
    #     time_vec = [0]
        
    #     while count < self.nsteps:
    #             t += dt
               
    #             count += 1
                
    #             u_sol.append(ODE.integrate(t))
    #             #dt = self.cfl_condition(u_sol[count])
    #             time_vec.append(t)
    #     return np.array(u_sol), np.array(time_vec)

    #### Helper functions for the model ####
        
    def _flux(self, u): ##No flux boundary conditios i.e. if j=0,....,2*L+1 then F_0-1/2 = 0, F_2*L+1 + 1/2 = 0
        """
        Calculates the numerical flux.
        """
        xi = self._potential_sum(u)
        vel = -(xi[1:] - xi[:-1]) / self.dx  # Velocity at cell interfaces

        velplus = jnp.maximum(0, vel)
        velminus = jnp.minimum(0, vel)

        u_east, u_west = self._reconstruct_east_west(u)

        # Calculate flux at internal interfaces
        internal_flux = velplus * u_east[:-1] + velminus * u_west[1:]
        
        # Create full flux array with no-flux boundary conditions (flux=0 at ends)
        flux = jnp.zeros(self.N + 1)
        flux = flux.at[1:-1].set(internal_flux)
        
        return flux

    def _potential_sum(self, u):
        """Calculates the interaction potential."""
        conv = self.dx * jnp.convolve(u, self._w_potential(self.x, u), mode="same")
        potential = self._h_energy(u) + conv
        return potential
    
    def _w_potential(self, x, u):
        """Interaction kernel potential."""
        return -self.aw * jnp.exp(-jnp.abs(x)**2 / (2 * self.sigma)) / jnp.sqrt(2 * jnp.pi * self.sigma)
    
    def _h_energy(self, u):
        """Internal energy term H'(u)."""
        return self.D * u**(self.m - 1)
    
    def _reconstruct_east_west(self, u):
        """
        Vectorized reconstruction of cell-edge values (u_east, u_west).
        This replaces the original for-loop and component-wise function.
        """
        # Pad u with zeros for boundary conditions
        u_padded = jnp.pad(u, (1, 1))
        
        # Get cell values and their neighbors for all cells at once
        uj = u_padded[1:-1]  # u
        ujminus = u_padded[:-2] # u_{j-1}
        ujplus = u_padded[2:]  # u_{j+1}

        # --- Standard second-order reconstruction ---
        u_xj = (ujplus - ujminus) / (2 * self.dx)
        u_east_init = uj + self.dx / 2 * u_xj
        u_west_init = uj - self.dx / 2 * u_xj
        
        # --- Slope-limited reconstruction (for positivity) ---
        # Calculate the three slope candidates for the minmod limiter
        slope_candidates = jnp.stack([
            2 * (uj - ujminus) / self.dx,
            (ujplus - ujminus) / (2 * self.dx),
            2 * (ujplus - uj) / self.dx
        ], axis=-1)
        
        u_xj_rec = self._vectorized_minmod(slope_candidates)
        u_east_rec = uj + self.dx / 2 * u_xj_rec
        u_west_rec = uj - self.dx / 2 * u_xj_rec

        # --- Combine based on positivity condition ---
        # Check if the initial reconstruction is non-physical (negative)
        needs_reconstruction = (u_east_init < 0) | (u_west_init < 0)
        
        # Use jnp.where to choose between initial and limited reconstruction
        u_east = jnp.where(needs_reconstruction, u_east_rec, u_east_init)
        u_west = jnp.where(needs_reconstruction, u_west_rec, u_west_init)

        return u_east, u_west

    def _vectorized_minmod(self, p_vectors):
        """
        Vectorized minmod function.
        p_vectors is an array of shape (N, 3), where each row is a vector of slopes.
        """
        signs = jnp.sign(p_vectors)
        all_pos = jnp.all(signs > 0, axis=1)
        all_neg = jnp.all(signs < 0, axis=1)
        
        # If all slopes are positive, return the minimum.
        # If all slopes are negative, return the maximum.
        # Otherwise, return zero.
        result = jnp.where(all_pos, jnp.min(p_vectors, axis=1), 0.0)
        result = jnp.where(all_neg, jnp.max(p_vectors, axis=1), result)
        
        return result

    def _discrete_laplacian_1d(self, u):
        """Calculates the 1D discrete Laplacian with Neumann boundary conditions."""
        padded_u = jnp.pad(u, (1, 1), mode='edge') # Pad with edge values for Neumann B.C.
        lap = (padded_u[2:] - 2 * padded_u[1:-1] + padded_u[:-2]) / self.dx**2
        return lap

    def _growth_term(self, u):
        """Logistic growth term."""
        return self.r * u * (1 - (u / self.K))
    
    ##### OLD CODE #####
    
    def u_east_and_west_component(self, uj, ujminus, ujplus): #--Calculates u east and west COMPONENT WISE (j wise) so the recalculation of the slope can be computed easilty if needed
        
        u_xj = (ujplus - ujminus) / (2*self.dx)
        u_eastj = uj + self.dx/2 * u_xj
        u_westj = uj - self.dx/2 * u_xj
        
        if (u_eastj < 0) or (u_westj < 0):
            theta = 2
            vec = jnp.array([theta*(uj - ujminus)/self.dx, (ujplus - ujminus) / (2*self.dx), theta*(ujplus - uj)/self.dx])
            u_xj_rec = self.minmod(vec) 
            u_eastj = uj + self.dx/2 * u_xj_rec
            u_westj = uj - self.dx/2 * u_xj_rec
        
        return u_eastj, u_westj

    def u_east_and_west(self, u): #--Fills out eat and west vectors from the component function. u^west_0 and u^east_N are 0
    
        
        u_east = np.zeros(u.shape[0])
        u_west = np.zeros(u.shape[0])
        
        for j in range(1, self.N-1):
            u_east[j], u_west[j] = self.u_east_and_west_component(u[j], u[j-1], u[j+1])
        
        u_east[0], u_west[0] = self.u_east_and_west_component(u[0], 0, u[1])   
        
        u_east[self.N-1], u_west[self.N-1] = self.u_east_and_west_component(u[self.N-1], u[self.N-2], 0) 

        u_east = jnp.array(u_east)
        u_west = jnp.array(u_west)
    
        return u_east, u_west


    def minmod(self, p): #--Calculates minmod for a certain vector p
        if all(x > 0 for x in p) == True: return jnp.min(p)
        
        elif all(x < 0 for x in p) == True: return jnp.max(p)
        
        else: return 0

    def discrete_laplacian_1D(self, u): #--Discrete laplacian for when diffusion exponent m=1
        
        lap = jnp.zeros(self.N)
        
        lap[1:-1] = (u[2:] + u[:-2] - 2*u[1:-1]) / self.dx**2   
        lap[0] = (u[1] - u[0]) / self.dx**2
        lap[-1] = (u[-2] - u[-1]) / self.dx**2
        return lap
    
    def growth_term(self, u): #--Growth function, more can be added
        return self.r*u*(1- (u/self.K))
    
    
    def cfl_condition(self, u):
        
        eta = self.potential_sum(u)
        
        vel = self.vel(eta)
        
        velplus = jnp.maximum(0, vel)
        velminus = jnp.minimum(0, vel)
        a = 2 * jnp.max(jnp.maximum(velplus, - velminus))
        
        return self.dx/(a * 5)