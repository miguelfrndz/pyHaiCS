import sys
import unittest
from pathlib import Path

import jax
import jax.numpy as jnp

PYHAICS_PATH = Path(__file__).parents[2]
sys.path.append(str(PYHAICS_PATH))
sys.path.insert(0, str(PYHAICS_PATH))

import pyHaiCS as haics
from pyHaiCS.utils.test import HiddenPrints
from pyHaiCS.utils import hamiltonian


class TestHamiltonianUtils(unittest.TestCase):
    """
    Unit tests for functions in pyHaiCS.utils.hamiltonian
    - Tests kinetic energy computation
    - Tests Hamiltonian and RMHMC Hamiltonian functions
    - Validates Fisher metric and generalized Fisher metric constructions
    """

    def setUp(self):
        self.mass_matrix = jnp.eye(2)
        self.p = jnp.array([1.0, 2.0])
        self.x = jnp.array([0.5, -1.5])
        self.potential = lambda x: 0.5 * jnp.sum(x ** 2)
        self.metric = lambda x: jnp.eye(len(x)) * (1.0 + jnp.sum(x**2))

    @HiddenPrints
    def test_kinetic_energy(self):
        result = hamiltonian.Kinetic(self.p, self.mass_matrix)
        expected = 0.5 * jnp.dot(self.p, self.p)
        self.assertAlmostEqual(float(result), float(expected), places=5)

    @HiddenPrints
    def test_hamiltonian(self):
        result = hamiltonian.Hamiltonian(self.x, self.p, self.potential, self.mass_matrix)
        expected = self.potential(self.x) + 0.5 * jnp.dot(self.p, self.p)
        self.assertAlmostEqual(float(result), float(expected), places=5)

    @HiddenPrints
    def test_rmhmc_hamiltonian(self):
        result = hamiltonian.Hamiltonian_RMHMC(self.x, self.p, self.potential, self.metric)
        G = self.metric(self.x)
        K = 0.5 * jnp.dot(self.p, jnp.linalg.solve(G, self.p)) + 0.5 * jnp.linalg.slogdet(G)[1]
        expected = self.potential(self.x) + K
        self.assertAlmostEqual(float(result), float(expected), places=5)

    @HiddenPrints
    def test_fisher_metric(self):
        def log_likelihood(theta, data):
            return -0.5 * jnp.sum((data - theta) ** 2)

        data = jnp.array([1.0, 2.0])
        fisher_fn = hamiltonian.fisher_metric(log_likelihood, (data,))
        G = fisher_fn(jnp.array([1.0, 2.0]))
        self.assertEqual(G.shape, (2, 2))
        self.assertTrue(jnp.all(jnp.linalg.eigvals(G) > 0))  # Positive definite

    @HiddenPrints
    def test_generalized_fisher_metric(self):
        def log_likelihood(theta, data):
            return -0.5 * jnp.sum((data - theta) ** 2)

        def neg_log_prior(theta, prior_scale):
            return 0.5 * jnp.sum((theta / prior_scale) ** 2)

        data = jnp.array([1.0, 2.0])
        prior_scale = jnp.array([1.0, 1.0])
        gfisher_fn = hamiltonian.generalized_fisher_metric(
            log_likelihood_fn=log_likelihood,
            neg_log_prior_fn=neg_log_prior,
            log_likelihood_params=(data,),
            prior_params=(prior_scale,)
        )
        G = gfisher_fn(jnp.array([0.5, -1.0]))
        self.assertEqual(G.shape, (2, 2))
        self.assertTrue(jnp.all(jnp.linalg.eigvals(G) > 0))  # Positive definite

if __name__ == '__main__':
    unittest.main()
