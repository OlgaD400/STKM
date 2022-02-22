import unittest
from TKM import simplex_prox, TKM
import numpy as np


class Tests(unittest.TestCase):
    """
    Test class for tkm
    """

    def test_simplex(self):
        """
        Test that the simplex prox projects onto the capped simplex.
        """

        a = np.random.rand(3, 4, 3)

        for num in range(5):
            proj = simplex_prox(a, num)

        assert np.all(np.isclose(np.sum(proj, axis=2), num))



