import jax
import jax.numpy as jnp
from functools import partial
from jax.scipy.linalg import cho_solve

class MassMatrix:
    """M matrix, his Cholesky factor and its inverse, precomputed."""
    def __init__(self, M, chol, Minv):
        self.M, self.chol, self.Minv = M, chol, Minv

    @classmethod
    def from_matrix(cls, M):
        M = jnp.asarray(M)
        chol = jnp.linalg.cholesky(M)
        Minv = cho_solve((chol, True), jnp.eye(M.shape[0], dtype=M.dtype))
        return cls(M, chol, Minv)

    def sample(self, key):              # momentum with M as covariance
        return self.chol @ jax.random.normal(key, (self.M.shape[0],), self.M.dtype)

    def velocity(self, p):              # computes M^-1 * p
        return self.Minv @ p

    def kinetic(self, p):
        return 0.5 * p @ (self.Minv @ p)
    
jax.tree_util.register_pytree_node(
    MassMatrix,
    lambda m: ((m.M, m.chol, m.Minv), None),
    lambda _, c: MassMatrix(*c))

def as_mass(M):
    return M if isinstance(M, MassMatrix) else MassMatrix.from_matrix(M)
    
@jax.jit
def Kinetic(p, mass_matrix):
    """
    Kinetic energy function.
    -------------------------
    Parameters:
        p (jax.Array): momentum
        mass_matrix (jax.Array or MassMatrix): mass matrix
    -------------------------
    Returns:
        K (float): kinetic energy
    """
    return as_mass(mass_matrix).kinetic(p)

@partial(jax.jit, static_argnums=(2, ))
def Hamiltonian(x, p, potential, mass_matrix):
    """
    Hamiltonian function.
    -------------------------
    Parameters:
        x (jax.Array): position
        p (jax.Array): momentum
        potential (function): Hamiltonian potential
        mass_matrix (jax.Array or MassMatrix): mass matrix
    -------------------------
    Returns:
        H (float): Hamiltonian
    """
    K = Kinetic(p, mass_matrix)
    U = potential(x)
    return U + K

def Extended_Hamiltonian(modified_hamiltonian, mu, mass_matrix):
    """
    Extended Hamiltonian for Modified Hamiltonian Monte Carlo.
    -------------------------
    Parameters:
        modified_hamiltonian (float): modified Hamiltonian value
        mu (jax.Array): noise vector
        mass_matrix (jax.Array or MassMatrix): mass matrix
    -------------------------
    Returns:
        H_ext (float): Extended Hamiltonian
    """
    return modified_hamiltonian + Kinetic(mu, mass_matrix)

def Position_Terms(x, potential, potential_grad, potential_hessian):
    """
    Terms of the Modified Hamiltonian that depend only on the position.
    -------------------------
    Parameters:
        x (jax.Array): position
        potential (function): Hamiltonian potential
        potential_grad (function): gradient of the Hamiltonian potential
        potential_hessian (function): Hessian of the Hamiltonian potential
    -------------------------
    Returns:
        terms (tuple): (U, grad_U, hess_U)
    """
    return potential(x), potential_grad(x), potential_hessian(x)

def Modified_Hamiltonian_From_Terms(x, p, terms, potential_grad, mass_matrix, step_size, order, c, M_inv=None):
    """
    Modified Hamiltonian (4th or 6th order) for MHMC, given the position terms.
    Same value as Modified_Hamiltonian, but U, gradient and Hessian are not recomputed:
    they can be reused for every momentum evaluated at the same position.
    -------------------------
    Parameters:
        x (jax.Array): position (used only by the 6th order term, through potential_grad)
        p (jax.Array): momentum
        terms (tuple): (U, grad_U, hess_U), as returned by Position_Terms
        potential_grad (function): gradient of the Hamiltonian potential (6th order only)
        mass_matrix (jax.Array or MassMatrix): mass matrix
        step_size (float): step size for the integration
        order (int): order of the method (4 or 6)
        c (dict): coefficients for the correction terms {c21, c22, ...}
        M_inv (jax.Array or None): inverse of the mass matrix, computed here if None
    -------------------------
    Returns:
        H (float): Modified Hamiltonian
    """
    U, grad_U, hess_U = terms
    if M_inv is None:
        M_inv = as_mass(mass_matrix).Minv
    K = Kinetic(p, mass_matrix)
    H = K + U

    v = M_inv @ p               # M^{-1} p
    g_M = M_inv @ grad_U        # M^{-1} grad(U)
    Hv = hess_U @ v             # H(U) v   

    corr = step_size**2 * (
        c['c21'] * v @ Hv +
        c['c22'] * grad_U @ g_M
    )

    if order == 6:
        # Third-order derivative contraction: U_xxx[v, v]
        U3_contr = jax.jvp(lambda y: jax.jvp(potential_grad, (y,), (v,))[1], (x,), (v,))[1]
        Hg_M = hess_U @ g_M

        term1 = c['c41'] * v @ U3_contr        # U_xxx[v, v, v]
        term2 = c['c42'] * v @ Hg_M            # p^T M^-1 H M^-1 grad U
        term3 = c['c43'] * g_M @ Hg_M          # grad U^T M^-1 H M^-1 grad U
        term4 = c['c44'] * Hv @ (M_inv @ Hv)   # p^T M^-1 H M^-1 H M^-1 p

        corr += step_size**4 * (term1 + term2 + term3 + term4)

    return H + corr

def Modified_Hamiltonian(x, p, potential, potential_grad, potential_hessian, mass_matrix, step_size, order, c):
    """
    Modified Hamiltonian (4th or 6th order) for MHMC.
    -------------------------
    Parameters:
        x (jax.Array): position
        p (jax.Array): momentum
        potential (function): Hamiltonian potential
        potential_grad (function): gradient of the Hamiltonian potential
        potential_hessian (function): Hessian of the Hamiltonian potential
        mass_matrix (jax.Array or MassMatrix): mass matrix
        step_size (float): step size for the integration
        order (int): order of the method (4 or 6)
        c (dict): coefficients for the correction terms {c21, c22, ...}
    -------------------------
    Returns:
        H (float): Modified Hamiltonian
    """
    terms = Position_Terms(x, potential, potential_grad, potential_hessian)
    mass = as_mass(mass_matrix)
    return Modified_Hamiltonian_From_Terms(x, p, terms, potential_grad, mass, step_size, order, c)

@partial(jax.jit, static_argnums=(2, 3))
def Hamiltonian_RMHMC(x, p, potential, metric):
    """
    Riemannian Manifold Hamiltonian.
    -------------------------
    Parameters:
        x (jax.Array): position
        p (jax.Array): momentum
        potential (function): potential energy function
        metric (function): function that returns metric tensor G(x)
    -------------------------
    Returns:
        H (float): Non-Separable Hamiltonian with position-dependent metric
    """
    U = potential(x)
    G = metric(x)
    K = 0.5 * jnp.dot(p, jnp.linalg.solve(G, p)) + 0.5 * jnp.linalg.slogdet(G)[1]
    return U + K

def fisher_metric(log_likelihood_fn, log_likelihood_params):
    """
    Returns a function G(theta) that computes the observed Fisher Information:
        G(θ) = ∇θ log p(data | θ) ∇θ log p(data | θ)^T
    -------------------------
    Parameters:
        log_likelihood_fn (function): log-likelihood function
        log_likelihood_params (tuple): parameters for the log-likelihood function
    -------------------------
    Returns:
        G_fn (function): Fisher information metric function G(θ)
    -------------------------
    """
    log_likelihood_fn = jax.jit(jax.tree_util.Partial(log_likelihood_fn, *log_likelihood_params))
    grad_loglik_fn = jax.grad(lambda t: log_likelihood_fn(t))
    def G_fn(theta):
        grad_loglik = grad_loglik_fn(theta)
        fisher_part = jnp.outer(grad_loglik, grad_loglik)
        return fisher_part + 1e-4 * jnp.eye(theta.shape[0])
    return jax.jit(G_fn)

def generalized_fisher_metric(log_likelihood_fn, neg_log_prior_fn, log_likelihood_params, prior_params):
    """
    Returns a function G(theta) that computes the Generalized Fisher Information Metric:
        G(θ) = ∇log p(data|θ)∇log p(data|θ)ᵀ + H(-log p(θ))
    -------------------------
    Parameters:
        log_likelihood_fn (function): log-likelihood function
        neg_log_prior_fn (function): negative log-prior function
        log_likelihood_params (tuple): parameters for the log-likelihood function
        prior_params (tuple): parameters for the negative log-prior function
    -------------------------
    Returns:
        G_fn (function): Generalized Fisher information metric function G(θ)
    -------------------------
    """
    log_likelihood_fn = jax.jit(jax.tree_util.Partial(log_likelihood_fn, *log_likelihood_params))
    grad_loglik_fn = jax.grad(lambda t: log_likelihood_fn(t))
    neg_log_prior_fn = jax.jit(jax.tree_util.Partial(neg_log_prior_fn, *prior_params))
    hessian_prior_fn = jax.hessian(lambda t: neg_log_prior_fn(t))
    def G_fn(theta):
        # Observed Fisher Metric from Likelihood
        grad_loglik = grad_loglik_fn(theta)
        fisher_part = jnp.outer(grad_loglik, grad_loglik)
        # Hessian from Negative Log-Prior
        prior_part = hessian_prior_fn(theta)
        # Return the combined metric, with jitter for stability
        return fisher_part + prior_part + 1e-4 * jnp.eye(theta.shape[0])
    return jax.jit(G_fn)