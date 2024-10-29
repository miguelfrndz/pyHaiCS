import sys, unittest
from pathlib import Path

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

if __name__ == '__main__':
    unittest.main()