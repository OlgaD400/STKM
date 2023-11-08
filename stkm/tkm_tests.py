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
    
    def test_cosine_similarity(self):
        """Test cos(xti, ctj) and cos(ctj, ct+1j)"""
        timesteps = 10
        dimension = 3
        num_points = 100
        num_clusters = 2

        data_centers_cos_sim = np.zeros((timesteps, num_points, num_clusters))
        centers_cos_sim = np.zeros((timesteps, num_clusters))

        data = np.random.rand(timesteps,dimension,num_points)
        centers = np.random.rand(timesteps,dimension,num_clusters)
        centers_shifted = np.vstack(
            (centers[0, :, :][np.newaxis, :, :], centers[:-1, :, :])
        )

        for t in range(timesteps):
            for j in range(num_clusters):
                for i in range(num_points):
                    data_centers_cos_sim[t,i,j] = cosine_similarity(data[t, : , i][np.newaxis, ...], centers[t,:, j][np.newaxis, ...])[0]
                
                centers_cos_sim[t,j] = cosine_similarity(centers[t, : , j][np.newaxis, ...], centers_shifted[t,:, j][np.newaxis, ...])[0]
    
        tkm = STKM(data)
        tkm_data_centers_cos_sim,_ = tkm.cosine_similarity_data_centers(data = data, centers = centers)
        tkm_centers_cos_sim,_ = tkm.cosine_similarity_centers(centers= centers, centers_shifted = centers_shifted)

        assert np.all(np.isclose(data_centers_cos_sim, tkm_data_centers_cos_sim))
        assert np.all(np.isclose(centers_cos_sim, tkm_centers_cos_sim))

    def test_center_data_derivative(self):
        """Test derivative of cos(xti, ctj)"""
        timesteps = 10
        dimension = 3
        num_points = 100
        num_clusters = 2

        data = np.random.rand(timesteps,dimension,num_points)
        weights = np.random.rand(timesteps,num_points,num_clusters)
        centers = np.random.rand(timesteps,dimension,num_clusters)

        #txnxk
        derivative = np.zeros((timesteps, dimension, num_clusters))

        for t in range(timesteps):
            for j in range(num_clusters):
                curr_sum = np.zeros((1,dimension))
                for i in range(num_points): 
                    center_norm = np.linalg.norm(centers[t, :, j],2)
                    data_norm = np.linalg.norm(data[t, :, i], 2)
                    term_1_denominator = center_norm*data_norm

                    term_1 = (weights[t, i, j] * data[t,:,i])/term_1_denominator

                    cos_sim_xc = cosine_similarity(data[t, : , i][np.newaxis, ...], centers[t,:, j][np.newaxis, ...])[0]
                    term_2_quotient = centers[t,:,j]/center_norm**2
                    term_2 = weights[t,i,j] * cos_sim_xc * term_2_quotient
                    
                    term_difference = term_1 - term_2
                    curr_sum += term_difference
                    
                derivative[t, :, j] = curr_sum

        tkm = STKM(data = data)
        tkm_derivative = tkm.cosine_similarity_derivative_data_centers(data = data, centers = centers, weights = weights)

        assert np.all(np.isclose(derivative, tkm_derivative))

    def test_center_derivative(self):
        """Test derivative (ctj, ct+1j)"""

        timesteps = 10
        dimension = 3
        num_points = 100
        num_clusters = 2
        lam = .80

        data = np.random.rand(timesteps,dimension,num_points)
        centers = np.random.rand(timesteps,dimension,num_clusters)
        centers_shifted = np.vstack(
            (centers[0, :, :][np.newaxis, :, :], centers[:-1, :, :])
        )

        #txnxk
        derivative = np.zeros((timesteps, dimension, num_clusters))

        for t in range(timesteps):
            for j in range(num_clusters):
                center_norm = np.linalg.norm(centers[t, :, j],2)
                center_shifted_norm = np.linalg.norm(centers_shifted[t,:,j])
                term_1_denominator = center_norm*center_shifted_norm

                term_1 = (centers_shifted[t,:,j])/term_1_denominator

                cos_sim_xc = cosine_similarity(centers[t, : , j][np.newaxis, ...], centers_shifted[t,:, j][np.newaxis, ...])[0]
                term_2_quotient = centers[t,:,j]/center_norm**2
                term_2 = cos_sim_xc*term_2_quotient
                
                term_difference = lam*num_points*(term_1 - term_2)
                
                derivative[t, :, j] = term_difference

        tkm = STKM(data = data)
        tkm_derivative = tkm.cosine_similarity_derivative_centers(centers = centers, centers_shifted=centers_shifted, lam = lam, num_points = num_points)

        assert np.all(np.isclose(derivative, tkm_derivative))

    def run_tests(self):
        """Run all tests"""
        self.test_center_data_derivative()
        self.test_center_derivative()
        self.test_cosine_similarity()

tests = Tests()
tests.run_tests()