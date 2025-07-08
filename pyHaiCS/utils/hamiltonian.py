import jax
import jax.numpy as jnp

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
    K = 0.5 * jnp.dot(p, jnp.linalg.solve(mass_matrix, p))
    return K

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
    H = K + U
    return H