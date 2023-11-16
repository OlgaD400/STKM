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

    def test_run_stkm(self):
        """Test STkM initialization."""
        data = np.random.rand(10, 3, 100)
        stkm = STKM(data=data)
        self.assertRaises(AssertionError, stkm.perform_clustering, num_clusters=101)

    def test_data_center_l1_derivative(self):
        """Test ||x_ti - ctj||_1 derivative"""
        timesteps = 10
        num_dimensions = 3
        num_clusters = 2
        num_points = 20

        centers = np.zeros((timesteps, num_dimensions, num_clusters))
        larger_data = np.ones((timesteps, num_dimensions, num_points))
        weights = np.random.rand(timesteps, num_points, num_clusters)

        stkm = STKM(data=larger_data)
        data_center_derivative = stkm.data_center_l1_derivative(
            centers=centers, weights=weights
        )

        weight_sum = np.sum(weights, axis=1)[:, np.newaxis, :]
        target_derivative = np.repeat(weight_sum, num_dimensions, axis=1)

        assert np.all(
            data_center_derivative == target_derivative
        ), "Incorrect derivative."

        centers = np.ones((timesteps, num_dimensions, num_clusters))
        smaller_data = np.zeros((timesteps, num_dimensions, num_points))
        stkm = STKM(data=smaller_data)
        data_center_derivative_negative = stkm.data_center_l1_derivative(
            centers=centers, weights=weights
        )

        assert np.all(
            data_center_derivative_negative == -target_derivative
        ), "Incorrect derivative.s"

    def run_tests(self):
        """Run all tests"""
        self.test_simplex()
        self.test_run_stkm()
        self.test_data_center_l1_derivative()


tests = Tests()
tests.run_tests()
