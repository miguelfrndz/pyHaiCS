import unittest
import jax
import jax.numpy as jnp

from pyHaiCS.samplers.adaptive import sAIA, optimal_momentum_noise, _sAIA_OptimalCoeffs, sMAIA
from pyHaiCS.integrators.integrators import ME_2, VV_2, ME_3, VV_3
from pyHaiCS.utils.test import HiddenPrints

class TestSAIA(unittest.TestCase):
    @staticmethod
    def quadratic_potential(x):
        return 0.5 * jnp.sum(x**2)

    def setUp(self):
        self.x_init = jnp.array([0.0])
        self.potential_args = ()
        self.n_samples_tune = 10
        self.n_samples_check = 5
        self.n_samples_burn_in = 10
        self.n_samples_prod = 10
        self.mass_matrix = jnp.eye(1)
        self.RNG_key = 42

    @HiddenPrints
    def test_sAIA_HMC_output_shape(self):
        samples = sAIA(
            x_init=self.x_init,
            potential_args=self.potential_args,
            n_samples_tune=self.n_samples_tune,
            n_samples_check=self.n_samples_check,
            n_samples_burn_in=self.n_samples_burn_in,
            n_samples_prod=self.n_samples_prod,
            potential=self.quadratic_potential,
            mass_matrix=self.mass_matrix,
            sampler="HMC",
            stage=2,
            RNG_key=self.RNG_key
        )
        self.assertEqual(samples.shape, (self.n_samples_prod, 1))

    @HiddenPrints
    def test_sAIA_GHMC_output_shape(self):
        samples = sAIA(
            x_init=self.x_init,
            potential_args=self.potential_args,
            n_samples_tune=self.n_samples_tune,
            n_samples_check=self.n_samples_check,
            n_samples_burn_in=self.n_samples_burn_in,
            n_samples_prod=self.n_samples_prod,
            potential=self.quadratic_potential,
            mass_matrix=self.mass_matrix,
            sampler="GHMC",
            stage=2,
            RNG_key=self.RNG_key
        )
        self.assertEqual(samples.shape, (self.n_samples_prod, 1))

    @HiddenPrints
    def test_sMAIA_MHMC_output_shape(self):
        params_samples, params_weights = sMAIA(
            x_init=self.x_init,
            potential_args=self.potential_args,
            n_samples_tune=self.n_samples_tune,
            n_samples_check=self.n_samples_check,
            n_samples_burn_in=self.n_samples_burn_in,
            n_samples_prod=self.n_samples_prod,
            potential=self.quadratic_potential,
            mass_matrix=self.mass_matrix,
            stage=2,
            RNG_key=self.RNG_key
        )
        self.assertEqual(params_samples.shape, (self.n_samples_prod, 1))
        self.assertEqual(params_weights.shape, (self.n_samples_prod,))

    @HiddenPrints
    def test_invalid_stage_raises(self):
        with self.assertRaises(NotImplementedError):
            sAIA(
                x_init=self.x_init,
                potential_args=self.potential_args,
                n_samples_tune=self.n_samples_tune,
                n_samples_check=self.n_samples_check,
                n_samples_burn_in=self.n_samples_burn_in,
                n_samples_prod=self.n_samples_prod,
                potential=self.quadratic_potential,
                mass_matrix=self.mass_matrix,
                sampler="HMC",
                stage=4,  # Unsupported
                RNG_key=self.RNG_key
            )

    @HiddenPrints
    def test_invalid_sampler_raises(self):
        with self.assertRaises(NotImplementedError):
            sAIA(
                x_init=self.x_init,
                potential_args=self.potential_args,
                n_samples_tune=self.n_samples_tune,
                n_samples_check=self.n_samples_check,
                n_samples_burn_in=self.n_samples_burn_in,
                n_samples_prod=self.n_samples_prod,
                potential=self.quadratic_potential,
                mass_matrix=self.mass_matrix,
                sampler="L2MC",  # Unsupported
                stage=2,
                RNG_key=self.RNG_key
            )
    
    @HiddenPrints
    def test_stage_3_HMC_shape(self):
        samples = sAIA(
            x_init=self.x_init,
            potential_args=self.potential_args,
            n_samples_tune=self.n_samples_tune,
            n_samples_check=self.n_samples_check,
            n_samples_burn_in=self.n_samples_burn_in,
            n_samples_prod=self.n_samples_prod,
            potential=self.quadratic_potential,
            mass_matrix=self.mass_matrix,
            sampler="HMC",
            stage=3,
            RNG_key=self.RNG_key
        )
        self.assertEqual(samples.shape, (self.n_samples_prod, 1))

    @HiddenPrints
    def test_optimal_momentum_noise_bounds(self):
        dim = self.x_init.shape[0]
        step_size_nd = 1.5
        noise = optimal_momentum_noise(step_size_nd, stage=2, D=dim, a=0.0, b=1/4)
        self.assertTrue(0 <= noise <= 1)

    @HiddenPrints
    def test_optimal_coeffs_output_shape(self):
        # Use fake dimensionless step sizes
        dss = jnp.ones((5,))
        coeffs = _sAIA_OptimalCoeffs(dss, stage=3, key=0)
        self.assertEqual(coeffs.shape, (5,))
        # Check bounds
        b_min, b_max = ME_3().b, VV_3().b
        self.assertTrue(jnp.all((coeffs >= b_min) & (coeffs <= b_max)))

    @HiddenPrints
    def test_optimal_coeffs_track_stability_interval(self):
        coeffs = _sAIA_OptimalCoeffs(jnp.array([0.05, 3.95]), stage=2, key=0)
        self.assertLess(abs(float(coeffs[0] - ME_2().b)), 0.01)
        self.assertLess(abs(float(coeffs[1] - VV_2().b)), 0.01)

    @HiddenPrints
    def test_reproducibility(self):
        samples_1 = sAIA(
            x_init=self.x_init,
            potential_args=self.potential_args,
            n_samples_tune=5,
            n_samples_check=2,
            n_samples_burn_in=5,
            n_samples_prod=5,
            potential=self.quadratic_potential,
            mass_matrix=self.mass_matrix,
            sampler="HMC",
            stage=2,
            RNG_key=123
        )
        samples_2 = sAIA(
            x_init=self.x_init,
            potential_args=self.potential_args,
            n_samples_tune=5,
            n_samples_check=2,
            n_samples_burn_in=5,
            n_samples_prod=5,
            potential=self.quadratic_potential,
            mass_matrix=self.mass_matrix,
            sampler="HMC",
            stage=2,
            RNG_key=123
        )
        self.assertTrue(jnp.allclose(samples_1, samples_2))

    @HiddenPrints
    def test_invalid_optimal_coeffs_stage(self):
        with self.assertRaises(NotImplementedError):
            _sAIA_OptimalCoeffs(jnp.ones((5,)), stage=5, key=0)

if __name__ == "__main__":
    unittest.main()
