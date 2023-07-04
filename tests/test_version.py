import unittest

import minigrid_language_wrapper


class VersionTestCase(unittest.TestCase):
    """Version tests"""

    def test_version(self):
        """check minigrid_language_wrapper exposes a version attribute"""
        self.assertTrue(hasattr(minigrid_language_wrapper, "__version__"))
        self.assertIsInstance(minigrid_language_wrapper.__version__, str)


if __name__ == "__main__":
    unittest.main()
