import unittest
from stkm.STKM import STKM, simplex_prox
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class Tests(unittest.TestCase):
    def test_simplex(self):
        """
        Test that the simplex prox projects onto the capped simplex.
        """

        a = np.random.rand(3, 4, 3)

        for num in range(5):
            proj = simplex_prox(a, num)

        assert np.all(np.isclose(np.sum(proj, axis=2), num))
    
    def run_tests(self):
        """Run all tests"""
        self.test_simplex()

tests = Tests()
tests.run_tests()