import sys, unittest, jax
import jax.numpy as jnp
from pathlib import Path
from pyHaiCS.utils.test import HiddenPrints

PYHAICS_PATH = Path(__file__).parents[2]
sys.path.append(str(PYHAICS_PATH))

import pyHaiCS as haics

class TestImports(unittest.TestCase):

    def test_imports(self):
        self.assertTrue(haics.integrators.MSSI_2)
        self.assertTrue(haics.integrators.VV_2)
        self.assertTrue(haics.integrators.BCSS_2)
        self.assertTrue(haics.integrators.ME_2)
        self.assertTrue(haics.integrators.MSSI_3)
    
    def test_baseClass_nonImplmenetable(self):
        with self.assertRaises(NotImplementedError):
            haics.integrators.Integrator().integrate()

    def test_integrator_params(self):
        self.assertEqual(haics.integrators.VV_2().b, 1/4)
        self.assertEqual(haics.integrators.BCSS_2().b, 0.211781)
        self.assertEqual(haics.integrators.ME_2().b, 0.193183)
        self.assertEqual(haics.integrators.VV_3().a, 1/3)
        self.assertEqual(haics.integrators.VV_3().b, 1/6)
        self.assertEqual(haics.integrators.BCSS_3().a, 0.296195)
        self.assertEqual(haics.integrators.BCSS_3().b, 0.118880)
        self.assertEqual(haics.integrators.ME_3().a, 0.290486)
        self.assertEqual(haics.integrators.ME_3().b, 0.108991)

    def test_pytree_roundtrip(self):
        for integ in [haics.integrators.VV_2(), haics.integrators.ME_3(), haics.integrators.MSSI_2(0.2)]:
            leaves, treedef = jax.tree_util.tree_flatten(integ)
            back = jax.tree_util.tree_unflatten(treedef, leaves)
            self.assertIs(type(back), type(integ))
            self.assertEqual(back.__dict__, integ.__dict__)

    def test_integrate_matches_python_loop(self):
        grad = jax.grad(lambda x: 0.5 * jnp.sum(x**2))
        M = jnp.eye(3)
        x0, p0 = jnp.array([1., -2., 0.5]), jnp.array([0.3, 0.1, -0.2])
        integ = haics.integrators.ME_2()
        for n in (1, 4):
            x, p = x0, p0
            for _ in range(n):                       
                x, p = integ._integrator_step(x, p, grad, M, 0.3, integ.b)
            xn, pn = integ.integrate(x0, p0, grad, n, M, 0.3)
            self.assertTrue(jnp.allclose(x, xn, atol=1e-6))
            self.assertTrue(jnp.allclose(p, pn, atol=1e-6))

    def test_n_steps_traced_and_no_recompile(self):
        grad = jax.grad(lambda x: 0.5 * jnp.sum(x**2))
        M = jnp.eye(3)
        x0, p0 = jnp.array([1., -2., 0.5]), jnp.array([0.3, 0.1, -0.2])
        f = jax.jit(lambda integ, n: integ.integrate(x0, p0, grad, n, M, 0.3))
        f(haics.integrators.MSSI_2(0.2), 3)
        f(haics.integrators.MSSI_2(0.21), 5)         
        self.assertEqual(f._cache_size(), 1)

    @HiddenPrints
    def test_sAIA_integrator_in_HMC(self):
        """HMC with an sAIA_2 integrator samples a standard Gaussian with the right variance."""
        integ = haics.integrators.sAIA_2(c=1.0, key=jax.random.PRNGKey(0))
        samples = haics.samplers.hamiltonian.HMC(
            x_init=jnp.zeros(3), potential_args=(), n_samples=1500, burn_in=200,
            step_size=0.5, n_steps=3, potential=lambda x: 0.5 * jnp.sum(x**2),
            mass_matrix=jnp.eye(3), integrator=integ, n_chains=2)
        self.assertAlmostEqual(float(samples.var()), 1.0, delta=0.15)         

if __name__ == '__main__':
    unittest.main()