import jax
import jax.numpy as jnp
from functools import partial

@jax.jit
def Kinetic(p, mass_matrix):
    """
    Kinetic energy function.
    -------------------------
    Parameters:
        p (jax.Array): momentum
        mass_matrix (jax.Array): mass matrix
    -------------------------
    Returns:
        K (float): kinetic energy
    """
    return 0.5 * jnp.dot(p, jnp.linalg.solve(mass_matrix, p))

@partial(jax.jit, static_argnums=(2, ))
def Hamiltonian(x, p, potential, mass_matrix):
    """
    Hamiltonian function.
    -------------------------
    Parameters:
        x (jax.Array): position
        p (jax.Array): momentum
        potential (function): Hamiltonian potential
        mass_matrix (jax.Array): mass matrix
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
        mass_matrix (jax.Array): mass matrix
    -------------------------
    Returns:
        H_ext (float): Extended Hamiltonian
    """
    return modified_hamiltonian + Kinetic(mu, mass_matrix)

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
        mass_matrix (jax.Array): mass matrix
        step_size (float): step size for the integration
        order (int): order of the method (4 or 6)
        c (dict): coefficients for the correction terms {c21, c22, ...}
    -------------------------
    Returns:
        H (float): Modified Hamiltonian
    """
    M_inv = jnp.linalg.inv(mass_matrix)
    grad_U = potential_grad(x)
    hess_U = potential_hessian(x)
    K = Kinetic(p, mass_matrix)
    U = potential(x)
    H = K + U

    corr = step_size**2 * (
        c['c21'] * p.T @ M_inv @ hess_U @ M_inv @ p +
        c['c22'] * grad_U.T @ M_inv @ grad_U
    )

    if order == 6:
        # Approximate third-order derivative contraction: U_{xxx}[M_inv p, M_inv p]
        U3_contr = jax.jvp(lambda v: jax.jvp(potential_grad, (x,), (v,))[1], (M_inv @ p,), (M_inv @ p,))[1]

        term1 = c['c41'] * p.T @ M_inv @ U3_contr
        term2 = c['c42'] * p.T @ M_inv @ hess_U @ M_inv @ grad_U
        term3 = c['c43'] * grad_U.T @ M_inv @ hess_U @ M_inv @ grad_U
        term4 = c['c44'] * p.T @ M_inv @ hess_U @ M_inv @ hess_U @ M_inv @ p

        corr += step_size**4 * (term1 + term2 + term3 + term4)

    return H + corr

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