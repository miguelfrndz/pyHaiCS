import sys, unittest
from pathlib import Path

PYHAICS_PATH = Path(__file__).parents[2]
sys.path.append(str(PYHAICS_PATH))

import pyHaiCS as hcs

class TestImports(unittest.TestCase):

    def test_version(self):
        self.assertEqual(hcs.__version__, "0.0.1")

    def test_namespace(self):
        with self.assertRaises(AttributeError):
            # If __init__ hierarchy is correct this should not be possible
            hcs.hmc 

if __name__ == '__main__':
    unittest.main()