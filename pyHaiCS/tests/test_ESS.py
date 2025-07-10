import sys, unittest
from pathlib import Path

import jax
import jax.numpy as jnp

PYHAICS_PATH = Path(__file__).parents[2]
sys.path.append(str(PYHAICS_PATH))

import pyHaiCS as haics
from pyHaiCS.utils.metrics import geyerESS, multiESS, codaESS

class TestEffectiveSampleSizes(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Target and proposal distributions
        def target_pdf(x):
            return jnp.sin(2 * jnp.pi * x) + 1

        def proposal_pdf(x):
            return jnp.where((x >= 0) & (x <= 1), 1.0, 0.0)

        N_SAMPLES = 10_000
        N_CHAINS = 4
        k = 2.0  # envelope constant

        def rejection_sample(key):
            key1, key2 = jax.random.split(key)
            u = jax.random.uniform(key1, shape=(N_SAMPLES,))
            y = jax.random.uniform(key2, shape=(N_SAMPLES,))
            accept_prob = target_pdf(y) / (k * proposal_pdf(y))
            accepted = u < accept_prob
            samples = y[accepted]
            return samples

        # Store in class for reuse
        cls.N_SAMPLES = N_SAMPLES
        cls.N_CHAINS = N_CHAINS

        main_key = jax.random.PRNGKey(0)
        keys = jax.random.split(main_key, N_CHAINS)
        all_samples = list(map(rejection_sample, keys))

        min_len = min(s.shape[0] for s in all_samples)
        aligned_samples = jnp.stack([s[:min_len] for s in all_samples])
        cls.aligned_samples = aligned_samples.reshape(N_CHAINS, min_len, 1)

    def test_average_ess_near_half(self):
        ess = geyerESS(self.aligned_samples, normalize=False) / self.N_SAMPLES
        multi = multiESS(self.aligned_samples, normalize=False) / self.N_SAMPLES
        coda = codaESS(self.aligned_samples, normalize=False, method='monotone-sequence') / self.N_SAMPLES

        avg_ess = float(jnp.mean(ess))
        avg_multi = float(jnp.mean(multi))
        avg_coda = float(jnp.mean(coda))

        for value, name in [(avg_ess, "GeyerESS"), (avg_multi, "multiESS"), (avg_coda, "codaESS")]:
            with self.subTest(method=name):
                self.assertTrue(0.4 <= value <= 0.6,
                                msg=f"{name} average ESS {value:.3f} not in expected range [0.4, 0.6]")

    def test_geyer_ess_output_valid(self):
        ess = geyerESS(self.aligned_samples)
        self.assertTrue(jnp.all(ess > 0), "GeyerESS contains non-positive values.")
        self.assertEqual(ess.shape[-1], self.aligned_samples.shape[-1])

    def test_multi_ess_output_valid(self):
        ess = multiESS(self.aligned_samples)
        self.assertTrue(jnp.all(ess > 0), "multiESS contains non-positive values.")

    def test_coda_ess_output_valid(self):
        ess = codaESS(self.aligned_samples, method='monotone-sequence')
        self.assertTrue(jnp.all(ess > 0), "codaESS contains non-positive values.")
        self.assertEqual(ess.shape[-1], self.aligned_samples.shape[-1])

if __name__ == '__main__':
    unittest.main()
