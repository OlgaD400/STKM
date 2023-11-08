import unittest
from TKM import simplex_prox
import numpy as np
from stgkm.distance_functions import s_journey


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


    def test_temporal_graph_distance(self):
        """
        Test temporal graph distance
        """

        connectivity_matrix = np.array(
            [
                [[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0]],
                [[0, 1, 0, 0], [1, 0, 0, 1], [0, 0, 0, 1], [0, 1, 1, 0]],
                [[0, 0, 0, 1], [0, 0, 1, 1], [0, 1, 0, 0], [1, 1, 0, 0]],
                [[0, 1, 1, 0], [1, 0, 1, 0], [1, 1, 0, 0], [0, 0, 0, 0]],
            ]
        )

        t,n,n = connectivity_matrix.shape
        #Ensure test cases are symmetric
        for i in range(t):
            assert np.all(connectivity_matrix[i,:,:] == connectivity_matrix[i,:,:].T)

        distance_matrix = s_journey(connectivity_matrix)
        assert np.all(
            distance_matrix
            == np.array(
                [
                    [[0, 1, 4, 2], [1, 0, 1, 2], [2, 1, 0, 1], [3, 3, 1, 0]],
                    [[0, 1, 2, 2], [1, 0, 3, 1], [2, 2, 0, 1], [3, 1, 1, 0]],
                    [[0, np.inf, np.inf, 1], [2, 0, 1, 1], [2, 1, 0, np.inf], [1, 1, 2, 0]],
                    [
                        [0, 1, 1, np.inf],
                        [1, 0, 1, np.inf],
                        [1, 1, 0, np.inf],
                        [np.inf, np.inf, np.inf, 0],
                    ],
                ]
            )
        )

        connectivity_matrix = np.array(
            [
                [[0, 1, 0, 0], [1, 0, 1, 1], [0, 1, 1, 1], [0, 1, 1, 0]],
                [[1, 0, 1, 0], [0, 0, 1, 1], [1, 1, 0, 0], [0, 1, 0, 1]],
                [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]],
            ]
        )
        distance_matrix = s_journey(connectivity_matrix)
        assert np.all(
            distance_matrix
            == np.array(
                [
                    [[0, 1, 2, 2], [1, 0, 1, 1], [2, 1, 0, 1], [2, 1, 1, 0]],
                    [[0, 2, 1, 2], [np.inf, 0, 1, 1], [1, 1, 0, np.inf], [2, 1, 2, 0]],
                    [
                        [0, 1, np.inf, np.inf],
                        [1, 0, np.inf, np.inf],
                        [np.inf, np.inf, 0, 1],
                        [np.inf, np.inf, 1, 0],
                    ],
                ]
            )
        )
