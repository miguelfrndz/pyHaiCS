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