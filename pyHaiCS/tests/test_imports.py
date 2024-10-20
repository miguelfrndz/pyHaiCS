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
            # If __init__ hierarchy is correct this should not be possible
            haics.HMC
        with self.assertRaises(AttributeError):
            haics.utils.standard_monte_carlo
        with self.assertRaises(AttributeError):
            haics.estimators.standard_monte_carlo

    def test_integrator_implementation(self):
        with self.assertRaises(NotImplementedError):
            haics.integrators.Integrator().integrate()

if __name__ == '__main__':
    unittest.main()