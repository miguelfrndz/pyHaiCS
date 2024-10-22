import jax
import jax.numpy as jnp
from .integrator import Integrator

class VerletIntegrator(Integrator):

    def __init__(self):
        pass

    def _position_full_step(self, x, p, step_size, mass_matrix):
        return x + step_size * jnp.linalg.solve(mass_matrix, p)

    def _momentum_full_step(self, x, p, step_size, potential_grad, potential_args):
        return p - step_size * potential_grad(x, *potential_args)

    def _momentum_half_step(self, x, p, step_size, potential_grad, potential_args):
        return p - step_size/2 * potential_grad(x, *potential_args)
    
    def _verlet_step(self, x, p, step_size, potential_grad, potential_args, mass_matrix):
        # Full-Step update for position
        x = self._position_full_step(x, p, step_size, mass_matrix)
        # Full-Step update for momentum
        p = self._momentum_full_step(x, p, step_size, potential_grad, potential_args)
        return x, p
    
    def integrate(self, x, p, potential_grad, potential_args, n_steps, mass_matrix, step_size):
        """
        Verlet integration for Hamiltonian dynamics.
        -------------------------
        Parameters:
        x (jax.Array): position
        p (jax.Array): momentum
        potential_grad (function): potential gradient
        potential_args (tuple): arguments for potential gradient
        n_steps (int): number of integration steps
        mass_matrix (jax.Array): mass matrix
        step_size (float): step size
        -------------------------
        Returns:
        x (jax.Array): updated position
        p (jax.Array): updated momentum
        """
        # Half-Step update for momentum
        p = self._momentum_half_step(x, p, step_size, potential_grad, potential_args)
        for _ in range(n_steps - 1):
            x, p = self._verlet_step(x, p, step_size, potential_grad, potential_args, mass_matrix)
        # Full-Step update for position
        x = self._position_full_step(x, p, step_size, mass_matrix)
        # Half-Step update for momentum
        p = self._momentum_half_step(x, p, step_size, potential_grad, potential_args)
        return x, p

# FIXME: The following code DOES JIT-Compile, but it's not as fast as the previous one :(

# import jax
# import jax.numpy as jnp
# from .integrator import Integrator
# from functools import partial

# class VerletIntegrator(Integrator):

#     def __init__(self):
#         pass

#     def integrate(self, x, p, potential_grad, potential_args, n_steps, mass_matrix, step_size):
#         """
#         Verlet integration for Hamiltonian dynamics.
#         -------------------------
#         Parameters:
#         x (jax.Array): position
#         p (jax.Array): momentum
#         potential_grad (function): potential gradient
#         potential_args (tuple): arguments for potential gradient
#         n_steps (int): number of integration steps
#         mass_matrix (jax.Array): mass matrix
#         step_size (float): step size
#         -------------------------
#         Returns:
#         x (jax.Array): updated position
#         p (jax.Array): updated momentum
#         """
        
#         @jax.jit
#         def _position_full_step(x, p, step_size, mass_matrix):
#             return x + step_size * jnp.linalg.solve(mass_matrix, p)

#         @jax.jit
#         def _momentum_full_step(x, p, step_size):
#             return p - step_size * potential_grad(x, *potential_args)

#         @jax.jit
#         def _momentum_half_step(x, p, step_size):
#             return p - step_size / 2 * potential_grad(x, *potential_args)

#         @jax.jit
#         def _verlet_step(x, p, step_size, mass_matrix):
#             # Full-Step update for position
#             x = _position_full_step(x, p, step_size, mass_matrix)
#             # Full-Step update for momentum
#             p = _momentum_full_step(x, p, step_size)
#             return x, p

#         # Half-Step update for momentum
#         p = _momentum_half_step(x, p, step_size)
#         for _ in range(n_steps - 1):
#             x, p = _verlet_step(x, p, step_size, mass_matrix)
#         # Full-Step update for position
#         x = _position_full_step(x, p, step_size, mass_matrix)
#         # Half-Step update for momentum
#         p = _momentum_half_step(x, p, step_size)
#         return x, p