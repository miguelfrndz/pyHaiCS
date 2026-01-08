import jax
import jax.numpy as jnp
from functools import partial

class Integrator:
    """
    Base class for Hamiltonian Integrators.
    """
    def __init__(self):
        self.name = self.__class__.__name__
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
    
class RMHMCVerletIntegrator(Integrator):
    """
    Generalized Leapfrog Integrator for Riemannian Manifold HMC (RMHMC).
    """
    def __init__(self):
        super().__init__()

    @partial(jax.jit, static_argnums=(0, 3))
    def _integrator_step(self, x, p, hamiltonian, step_size):
        # Half-Step update for momentum
        p = p - 0.5 * step_size * jax.grad(hamiltonian, argnums=0)(x, p)
        # Full-Step update for position
        x = x + step_size * jax.grad(hamiltonian, argnums=1)(x, p)
        # Half-Step update for momentum
        p = p - 0.5 * step_size * jax.grad(hamiltonian, argnums=0)(x, p)
        return x, p

    def integrate(self, x, p, hamiltonian, n_steps, step_size):
        """
        Generalized Leapfrog for RMHMC.
        -------------------------
        Parameters:
            x (jax.Array): position
            p (jax.Array): momentum
            potential_fn (function): potential energy
            metric_fn (function): G(x)
            n_steps (int): integration steps
            step_size (float): step size
        -------------------------
        Returns:
            x (jax.Array), p (jax.Array)
        """
        for _ in range(n_steps):
            x, p = self._integrator_step(x, p, hamiltonian, step_size)
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
        raise NotImplementedError("Generalized integrator not implemented. \
                                  Please use either MSSI_2, MSSI_3 or any of the \
                                  specific instances of these methods implemented.")
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
    
def get_mhmc_coeffs(order, stage, b = None, a = None):
    """
    Compute coefficients for 4th or 6th order modified Hamiltonians
    for 2-stage or 3-stage integrators based on parameters b (or a, b).
    """
    if order not in [4, 6]:
        raise ValueError("Only 4th or 6th order supported")
    if stage not in [2, 3]:
        raise ValueError("Only 2-stage or 3-stage integrators supported")
    if order == 4 and (b is None or a is not None):
        raise ValueError("b must be provided for 4th order coefficients")
    if order == 6 and (b is None or a is None):
        raise ValueError("a and b must be provided for 6th order coefficients")
    if stage == 2:
        # 2-stage version
        c21 = 1/24 * (6 * b - 1)
        c22 = 1/12 * (6 * b ** 2 - 6 * b + 1)
        c41 = 1/5760 * (7 - 30 * b)
        c42 = 1/240 * (-10 * b ** 2 + 15 * b - 3)
        c43 = 1/120 * (-30 * b ** 3 + 35 * b ** 2 - 15 * b + 2)
        c44 = 1/240 * (20 * b ** 2 - 1)
    elif stage == 3:
        # 3-stage version
        c21 = 1/12 * (1 - 6 * a * (1 - a) * (1 - 2 * b))
        c22 = 1/24 * (6 * a * (1 - 2 * b) ** 2 - 1)
        c41 = 1/720 * (1 + 2 * (a - 1) * a * (8 + 31 * (a - 1) * a) * (1 - 2 * b) - 4 * b)
        c42 = 1/240 * (
            6 * a ** 3 * (1 - 2 * b) ** 2
            - a ** 2 * (19 - 116 * b + 36 * b ** 2 + 240 * b ** 3)
            + a * (27 - 208 * b + 308 * b ** 2)
            - 48 * b ** 2 + 48 * b - 7
        )
        c43 = 1/180 * (1 + 15 * a * (1 - 2 * b) * (-1 + 2 * a * (2 - 3 * b + a * (4 * b - 2))))
        c44 = 1/240 * (-1 + 20 * a * (1 - 2 * b) * (b + a * (1 + 6 * (b - 1) * b)))

    return {
        "c21": c21, "c22": c22,
        "c41": c41, "c42": c42, "c43": c43, "c44": c44
    }
        