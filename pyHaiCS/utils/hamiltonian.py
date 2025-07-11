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

def fisher_metric(log_likelihood_fn, data):
    """
    Returns a function G(theta) that computes the observed Fisher Information:
        G(θ) = ∇θ log p(data | θ) ∇θ log p(data | θ)^T

    Parameters:
        log_likelihood_fn (function): log p(data | θ), takes (params, data)
        data (any): observed dataset

    Returns:
        G_fn (function): Fisher information metric function G(θ)
    """
    def G_fn(theta):
        grads = jax.vmap(lambda y_i: jax.grad(lambda t: log_likelihood_fn(t, y_i))(theta))(data)
        return grads.T @ grads
    return jax.jit(G_fn)