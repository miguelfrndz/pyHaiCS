import sys, unittest
from pathlib import Path

PYHAICS_PATH = Path(__file__).parents[2]
sys.path.append(str(PYHAICS_PATH))

import pyHaiCS as haics

class TestImports(unittest.TestCase):

    def test_version(self):
        self.assertEqual(haics.__version__, "0.0.1")

    def test_namespace(self):
        with self.assertRaises(AttributeError):
            # If __init__ hierarchy is correct this should not be possible
            haics.HMC
        with self.assertRaises(AttributeError):
            haics.utils.standard_monte_carlo
        with self.assertRaises(AttributeError):
            haics.estimators.standard_monte_carlo

    def test_integrator_implementation(self):
        with self.assertRaises(NotImplementedError):
            haics.integrators.Integrator().integrate()

    def test_estimators_namespace(self):
        # estimators should not be directly accessible
        with self.assertRaises(AttributeError):
            _ = haics.estimators

    def test_import_submodules(self):
        self.assertTrue(hasattr(haics.utils, "test"))
        self.assertTrue(hasattr(haics.utils.test, "HiddenPrints"))

    def test_integrators_namespace(self):
        # integrators should be accessible
        self.assertTrue(hasattr(haics, "integrators"))

    def test_integrator_class_exists(self):
        # Integrator class should exist in integrators
        self.assertTrue(hasattr(haics.integrators, "Integrator"))

    def test_import_submodules(self):
        # Importing submodules directly from haics should fail
        with self.assertRaises(AttributeError):
            _ = haics.standard_monte_carlo

    def test_repr_and_str(self):
        # __repr__ and __str__ should exist for main package
        self.assertTrue(hasattr(haics, "__repr__"))
        self.assertTrue(hasattr(haics, "__str__"))

if __name__ == '__main__':
    unittest.main()