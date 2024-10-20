import jax.numpy as jnp
from .integrator import Integrator

class VerletIntegrator(Integrator):
    def __init__(self):
        pass

    def _position_full_step(self, x, p, step_size, mass_matrix):
        return x + step_size * jnp.linalg.solve(mass_matrix, p)

    def _momentum_full_step(self, x, p, step_size, potential_grad):
        return p - step_size * potential_grad(x)

    def _momentum_half_step(self, x, p, step_size, potential_grad):
        return p - step_size/2 * potential_grad(x)

    def _verlet_step(self, x, p, step_size, potential_grad, mass_matrix):
        # Full-Step update for position
        x = self._position_full_step(x, p, step_size, mass_matrix)
        # Full-Step update for momentum
        p = self._momentum_full_step(x, p, step_size, potential_grad)
        return x, p

    def integrate(self, x, p, potential_grad, n_steps, mass_matrix, step_size):
        """
        Verlet integration for Hamiltonian dynamics.
        -------------------------
        Parameters:
        x (jax.Array): position
        p (jax.Array): momentum
        potential_grad (function): potential gradient
        n_steps (int): number of steps
        mass_matrix (jax.Array): mass matrix
        step_size (float): step size
        -------------------------
        Returns:
        x (jax.Array): updated position
        p (jax.Array): updated momentum
        """
        # Half-Step update for momentum
        p = self._momentum_half_step(x, p, step_size, potential_grad)
        for _ in range(n_steps - 1):
            x, p = self._verlet_step(x, p, step_size, potential_grad, mass_matrix)
        # Full-Step update for position
        x = self._position_full_step(x, p, step_size, mass_matrix)
        # Half-Step update for momentum
        p = self._momentum_half_step(x, p, step_size, potential_grad)
        return x, p