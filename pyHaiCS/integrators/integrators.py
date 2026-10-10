import jax
import jax.numpy as jnp
from ..utils.hamiltonian import as_mass

class Integrator:
    """
    Base class for Hamiltonian Integrators.

    Registered as a JAX PyTree so an instance can be given to jax.jit, jax.vmap etc. as an argument.
    Attributes are of two types:
      - Dynamic (in _leaves): Numerical coefficients (e.g. `b` in MSSI_2, `a` and `b`
        in MSSI_3). Traced by JAX, so mutating values won't trigger recompilation,
        and multi-step arrays can be sliced directly inside `lax.scan`.
      - Static (auxiliary data): Everything else (`name`, `stage`, `n_coeff_samples`,
        etc.). These form the jit cache key, allowing instances with
        identical metadata to share compiled code.
    """
    # Names of the attributes that are arrays (the leaves of the pytree). A subclass overrides this
    # tuple to declare its coefficients. Empty by default: the integrator is then fully static (for example VV).
    _leaves = ()

    def __init__(self):
        self.name = self.__class__.__name__
        pass

    def __init_subclass__(cls,**kwargs):
        # Runs automatically whenever a subclass is defined, so every
        # integrator is registered as a JAX pytree without using a decorator for each one.
        super().__init_subclass__(**kwargs)
        jax.tree_util.register_pytree_node(cls, cls._tree_flatten, cls._tree_unflatten)
        pass
    
    def _tree_flatten(self):
        """Flattens the integrator into PyTree children and static metadata.

        Returns:
            children: Tuple of leaf values declared in `_leaves` (traced by JAX).
            aux: Deterministic (sorted) tuple of non-leaf `(name, value)` pairs.
                Values must be hashable for JIT caching.
        """                                                                    
        children = tuple(getattr(self, k) for k in self._leaves)                                # dynamic attributes
        aux = tuple(sorted((k, v) for k, v in self.__dict__.items() if k not in self._leaves))  # static ones
        return children, aux

    @classmethod
    def _tree_unflatten(cls, aux, children):
        """Reconstructs an instance from PyTree children and static metadata.

        Uses __new__ to bypass __init__, preventing unintended re-initialization
        when JAX reconstitutes the instance with tracers.
        """
        obj = object.__new__(cls)                    
        obj.__dict__.update(dict(aux))
        obj.__dict__.update(dict(zip(cls._leaves, children)))
        return obj

    def integrate(self, *args, **kwargs):
        raise NotImplementedError("Integrator subclasses should implement this method!")

class VerletIntegrator(Integrator):
    """
    Leapfrog/Modified 1-Stage Verlet Integrator.
    """
    def __init__(self):
        super().__init__()
    
    def _position_full_step(self, x, p, step_size, mass_matrix):
        return x + step_size * as_mass(mass_matrix).velocity(p)

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
        def body(_, xp):
            return self._verlet_step(xp[0], xp[1], step_size, potential_grad, mass_matrix)  
        x, p = jax.lax.fori_loop(0, n_steps - 1, body, (x, p))
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
        def body(_, xp):
          return self._integrator_step(xp[0], xp[1], hamiltonian, step_size)
        return jax.lax.fori_loop(0, n_steps, body, (x, p))

class MultiStageSplittingIntegrator(Integrator):
    def __init__(self, stage):
        self.stage = stage
        super().__init__()

    def _solution_flow_A(self, x, p, t, mass_matrix):
        """
        Solution Flow A:
            Phi_t^A(x, p) = (x + t * M^(-1) * p, p)
        """
        return (x + t * as_mass(mass_matrix).velocity(p), p)
    
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
    _leaves = ("b",)
    def __init__(self, b):
        self.b = b
        super().__init__(stage = 2)

    def _integrator_step(self, x, p, potential_grad, mass_matrix, step_size, b):
        x, p = self._solution_flow_B(x, p, step_size * b, potential_grad)
        x, p = self._solution_flow_A(x, p, step_size/2, mass_matrix)
        x, p = self._solution_flow_B(x, p, step_size * (1 - 2 * b), potential_grad)
        x, p = self._solution_flow_A(x, p, step_size/2, mass_matrix)
        x, p = self._solution_flow_B(x, p, step_size * b, potential_grad)
        return x, p

    def integrate(self, x, p, potential_grad, n_steps, mass_matrix, step_size):
        def body(_, xp):
            return self._integrator_step(xp[0], xp[1], potential_grad, mass_matrix, step_size, self.b)
        return jax.lax.fori_loop(0, n_steps, body, (x, p))

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
    _leaves = ("a", "b")
    def __init__(self, a, b):
        self.a, self.b = a, b
        super().__init__(stage = 3)

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
        def body(_, xp):
            return self._integrator_step(xp[0], xp[1], potential_grad, mass_matrix, step_size, self.a, self.b)
        return jax.lax.fori_loop(0, n_steps, body, (x, p))
    
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
    if b is None or (stage == 3 and a is None):
        raise ValueError("b must be provided for 2-stage coefficients, a and b for 3-stage ones")
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

@jax.jit
def _rho_2(step_size, b):
    numerator = step_size**4 * (2 * b**2 * (1/2 - b) * step_size**2 + 4 * b**2 - 6 * b + 1)**2
    denominator = 8 * (2 - b * step_size**2) * (2 - (1/2 - b) * step_size**2) * (1 - b * (1/2 - b) * step_size**2)
    return numerator / denominator

@jax.jit
def _rho_3(step_size, b):
    numerator = step_size**4 * (-3 * b**4 + 8 * b**3 - 19/4 * b**2 + b + b**2 * step_size**2 * (b**3 - 5/4 * b**2 + b/2 - 1/16) - 1/16)**2
    denominator = 2 * (3 * b - b * step_size**2 * (b - 1/4) - 1) * (1 - 3 * b - b * step_size**2 * (b - 1/2)**2) * (-9 * b**2 + 6 * b - step_size**2 * (b**3 - 5/4 * b**2 + b/2 - 1/16) - 1)
    return numerator / denominator

def _optimal_b(h_bar, key, rho, b_ME, b_VV, n_coeff_samples):
    """
    Optimal coefficient b for a single dimensionless step size.
    -------------------------
    Parameters:
        h_bar (float): dimensionless step size, upper end of the interval (0, h_bar)
        key (jax.Array): PRNG key, split into one key for the candidates b and one for the step sizes
        rho (function): rho(h, b), the quantity whose maximum over h is minimized
        b_ME (float): lower end of the interval for b (Minimum Error integrator)
        b_VV (float): upper end of the interval for b (Velocity Verlet)
        n_coeff_samples (int): number of candidates b, and of step sizes in (0, h_bar)
    -------------------------
    Returns:
        b (float): the candidate with the smallest maximum of rho
    """
    k_b, k_h = jax.random.split(key)
    b_vals = b_ME + (b_VV - b_ME) * jax.random.uniform(k_b, (n_coeff_samples,))
    h_vals = h_bar * jax.random.uniform(k_h, (n_coeff_samples,))
    max_rho = jax.vmap(lambda b: jnp.max(jax.vmap(rho, in_axes=(0, None))(h_vals, b)))(b_vals)
    return b_vals[jnp.argmin(max_rho)]

class sAIA_2(MultiStageSplittingIntegrator):
    """
    2-Stage s-AIA Integrator: the coefficient b is optimized for every step size.
    """
    _leaves = ("c", "key")      

    def __init__(self, c, key, n_coeff_samples=20):
        """
        Parameters:
            c (float or jax.Array): burn-in constant, so that h_bar = c * step_size
            key (jax.Array): PRNG key used to draw the candidate coefficients
            n_coeff_samples (int): number of candidate coefficients
        """
        super().__init__(stage=2)
        self.c, self.key, self.n_coeff_samples = c, key, n_coeff_samples

    def coefficients(self, step_size):
        """
        Optimal coefficient b for the given step size.
        -------------------------
        Parameters:
            step_size (float): step size
        -------------------------
        Returns:
            b (jax.Array): optimal coefficient of the 2-stage integrator
        """
        return _optimal_b(self.c * step_size, self.key, _rho_2, ME_2().b, VV_2().b,
                          self.n_coeff_samples)

    def integrate(self, x, p, potential_grad, n_steps, mass_matrix, step_size):
        """
        Integrate the Hamiltonian dynamics with the optimal 2-stage coefficient for step_size.
        -------------------------
        Parameters:
            x (jax.Array): position
            p (jax.Array): momentum
            potential_grad (function): potential gradient
            n_steps (int): number of integration steps (can be a traced value)
            mass_matrix (jax.Array): mass matrix
            step_size (float): step size
        -------------------------
        Returns:
            x (jax.Array): updated position
            p (jax.Array): updated momentum
        """
        return MSSI_2(self.coefficients(step_size)).integrate(
            x, p, potential_grad, n_steps, mass_matrix, step_size)

class sAIA_3(MultiStageSplittingIntegrator):
    """
    3-Stage s-AIA Integrator: the coefficients a and b are optimized for every step size.
    """
    _leaves = ("c", "key")

    def __init__(self, c, key, n_coeff_samples=20):
        """
        Parameters:
            c (float or jax.Array): burn-in constant, so that h_bar = c * step_size
            key (jax.Array): PRNG key used to draw the candidate coefficients
            n_coeff_samples (int): number of candidate coefficients
        """
        super().__init__(stage=3)
        self.c, self.key, self.n_coeff_samples = c, key, n_coeff_samples

    def coefficients(self, step_size):
        """
        Optimal coefficients for the given step size.
        -------------------------
        Parameters:
            step_size (float): step size
        -------------------------
        Returns:
            a (jax.Array): optimal coefficient a, obtained from b
            b (jax.Array): optimal coefficient b of the 3-stage integrator
        """
        b = _optimal_b(self.c * step_size, self.key, _rho_3, ME_3().b, VV_3().b,
                       self.n_coeff_samples)
        a = (1 - 2 * b) / (4 * (1 - 3 * b))
        return a, b

    def integrate(self, x, p, potential_grad, n_steps, mass_matrix, step_size):
        """
        Integrate the Hamiltonian dynamics with the optimal 3-stage coefficients for step_size.
        -------------------------
        Parameters:
            x (jax.Array): position
            p (jax.Array): momentum
            potential_grad (function): potential gradient
            n_steps (int): number of integration steps (can be a traced value)
            mass_matrix (jax.Array): mass matrix
            step_size (float): step size
        -------------------------
        Returns:
            x (jax.Array): updated position
            p (jax.Array): updated momentum
        """
        a, b = self.coefficients(step_size)
        return MSSI_3(a, b).integrate(x, p, potential_grad, n_steps, mass_matrix, step_size)