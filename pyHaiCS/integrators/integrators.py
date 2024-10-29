import jax
import jax.numpy as jnp
from functools import partial

class Integrator:
    """
    Base class for Hamiltonian Integrators.
    """
    def __init__(self):
        pass

    def integrate(self, *args, **kwargs):
        raise NotImplementedError("Integrator subclasses should implement this method!")

class VerletIntegrator(Integrator):
    """
    Leapfrog/Modified 1-Stage Verlet Integrator.
    """
    def __init__(self):
        super().__init__()
    
    @partial(jax.jit, static_argnums=(0,))
    def _position_full_step(self, x, p, step_size, mass_matrix):
        return x + step_size * jnp.linalg.solve(mass_matrix, p)

    @partial(jax.jit, static_argnums=(0, 4))
    def _momentum_full_step(self, x, p, step_size, potential_grad):
        return p - step_size * potential_grad(x)

    @partial(jax.jit, static_argnums=(0, 4))
    def _momentum_half_step(self, x, p, step_size, potential_grad):
        return p - step_size/2 * potential_grad(x)
    
    @partial(jax.jit, static_argnums=(0, 4))
    def _verlet_step(self, x, p, step_size, potential_grad, mass_matrix):
        # Full-Step update for position
        x = self._position_full_step(x, p, step_size, mass_matrix)
        # Full-Step update for momentum
        p = self._momentum_full_step(x, p, step_size, potential_grad)
        return x, p
    
    def integrate(self, x, p, potential_grad, n_steps, mass_matrix, step_size):
        """
        Verlet Integration for Hamiltonian dynamics.
        -------------------------
        Parameters:
        x (jax.Array): position
        p (jax.Array): momentum
        potential_grad (function): potential gradient
        n_steps (int): number of integration steps
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

class MultiStageSplittingIntegrator(Integrator):
    def __init__(self, stage):
        self.stage = stage
        super().__init__()

    @partial(jax.jit, static_argnums=(0,))
    def _solution_flow_A(self, x, p, t, mass_matrix):
        """
        Solution Flow A:
            Phi_t^A(x, p) = (x + t * M^(-1) * p, p)
        """
        return (x + t * jnp.linalg.solve(mass_matrix, p), p)
    
    @partial(jax.jit, static_argnums=(0, 4))
    def _solution_flow_B(self, x, p, t, potential_grad):
        """
        Solution Flow B:
            Phi_t^B(x, p) = (x, p - t * U_x(x))
        """
        return (x, p - t * potential_grad(x))

    def integrate(self, x, p, potential_grad, n_steps, mass_matrix, step_size):
        """
        Multi-Stage Splitting Integration for Hamiltonian dynamics.
        -------------------------
        Parameters:
        x (jax.Array): position
        p (jax.Array): momentum
        potential_grad (function): potential gradient
        n_steps (int): number of integration steps
        mass_matrix (jax.Array): mass matrix
        step_size (float): step size
        -------------------------
        Returns:
        x (jax.Array): updated position
        p (jax.Array): updated momentum
        """
        # TODO: Implement General Form of Multi-Stage Splitting Integrator
        raise NotImplementedError("Generalized integrator not implemented. Please use either MSSI_2, MSSI_3 or any of the specific instances of these methods implemented.")
        pass

class MSSI_2(MultiStageSplittingIntegrator):
    """
    2-Stage Multi-Stage Splitting Integrator.
    """
    def __init__(self, b):
        self.b = b
        super().__init__(stage = 2)

    @partial(jax.jit, static_argnums=(0, 3))
    def _integrator_step(self, x, p, potential_grad, mass_matrix, step_size, b):
        x, p = self._solution_flow_B(x, p, step_size * b, potential_grad)
        x, p = self._solution_flow_A(x, p, step_size/2, mass_matrix)
        x, p = self._solution_flow_B(x, p, step_size * (1 - 2 * b), potential_grad)
        x, p = self._solution_flow_A(x, p, step_size/2, mass_matrix)
        x, p = self._solution_flow_B(x, p, step_size * b, potential_grad)
        return x, p

    def integrate(self, x, p, potential_grad, n_steps, mass_matrix, step_size):
        b = self.b
        for _ in range(n_steps):
            x, p = self._integrator_step(x, p, potential_grad, mass_matrix, step_size, b)
        return x, p

class VV_2(MSSI_2):
    """
    2-Stage Velocity-Verlet Integrator.
    """
    def __init__(self):
        super().__init__(b = 1/4)
    
    def integrate(self, x, p, potential_grad, n_steps, mass_matrix, step_size):
        return super().integrate(x, p, potential_grad, n_steps, mass_matrix, step_size)

class BCSS_2(MSSI_2):
    """
    2-Stage BCSS Integrator.
    """
    def __init__(self):
        super().__init__(b = 0.211781)

    def integrate(self, x, p, potential_grad, n_steps, mass_matrix, step_size):
        return super().integrate(x, p, potential_grad, n_steps, mass_matrix, step_size)

class ME_2(MSSI_2):
    """
    2-Stage Minimum Error Integrator.
    """
    def __init__(self):
        super().__init__(b = 0.193183)

    def integrate(self, x, p, potential_grad, n_steps, mass_matrix, step_size):
        return super().integrate(x, p, potential_grad, n_steps, mass_matrix, step_size)
    
class MSSI_3(MultiStageSplittingIntegrator):
    """
    3-Stage Multi-Stage Splitting Integrator.
    """
    def __init__(self, a, b):
        self.a, self.b = a, b
        super().__init__(stage = 3)

    @partial(jax.jit, static_argnums=(0, 3))
    def _integrator_step(self, x, p, potential_grad, mass_matrix, step_size, a, b):
        x, p = self._solution_flow_B(x, p, step_size * b, potential_grad)
        x, p = self._solution_flow_A(x, p, step_size * a, mass_matrix)
        x, p = self._solution_flow_B(x, p, step_size * (1/2 - b), potential_grad)
        x, p = self._solution_flow_A(x, p, step_size * (1 - 2 * a), mass_matrix)
        x, p = self._solution_flow_B(x, p, step_size * (1/2 - b), potential_grad)
        x, p = self._solution_flow_A(x, p, step_size * a, mass_matrix)
        x, p = self._solution_flow_B(x, p, step_size * b, potential_grad)
        return x, p

    def integrate(self, x, p, potential_grad, n_steps, mass_matrix, step_size):
        a, b = self.a, self.b
        for _ in range(n_steps):
            x, p = self._integrator_step(x, p, potential_grad, mass_matrix, step_size, a, b)
        return x, p
    
class VV_3(MSSI_3):
    """
    3-Stage Velocity-Verlet Integrator.
    """
    def __init__(self):
        super().__init__(a = 1/3, b = 1/6)
    
    def integrate(self, x, p, potential_grad, n_steps, mass_matrix, step_size):
        return super().integrate(x, p, potential_grad, n_steps, mass_matrix, step_size)

class BCSS_3(MSSI_3):
    """
    3-Stage BCSS Integrator.
    """
    def __init__(self):
        super().__init__(a = 0.296195, b = 0.118880)

    def integrate(self, x, p, potential_grad, n_steps, mass_matrix, step_size):
        return super().integrate(x, p, potential_grad, n_steps, mass_matrix, step_size)

class ME_3(MSSI_3):
    """
    3-Stage Minimum Error Integrator.
    """
    def __init__(self):
        super().__init__(a = 0.290486, b = 0.108991)

    def integrate(self, x, p, potential_grad, n_steps, mass_matrix, step_size):
        return super().integrate(x, p, potential_grad, n_steps, mass_matrix, step_size)