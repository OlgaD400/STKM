"""Implementation of spatiotemporal k means."""

from typing import Union
import numpy as np
from sklearn.cluster import kmeans_plusplus


def simplex_prox(z: np.ndarray, a: int) -> np.ndarray:
    """
    Project onto simplex.

    Args:
        z (np.ndarray): Matrix to be projected onto the simplex. Matrix must
            be of size time_steps x number of particles x dimension.
        a (int): Simplex to be projected onto.

    Returns:
        (np.ndarray): Projection of matrix z onto the a simplex.
    """
    u = z.copy()
    u[..., ::-1].sort(axis=2)
    j = np.arange(u.shape[2])
    v = (a - np.cumsum(u, axis=2)) / (j + 1)
    i = np.repeat(j[None, :], u.shape[1], axis=0)
    rho = np.max(i * (u + v > 0), axis=2)
    lam = v[
        np.repeat(np.arange(u.shape[0]), u.shape[1]),
        np.tile(np.arange(u.shape[1]), u.shape[0]),
        rho.flatten(),
    ].reshape(u.shape[0], u.shape[1], 1)
    return np.maximum(z + lam, 0.0)


class STKM:
    """
    Create a class to perform Time K Means Clustering.
    """

    def __init__(self, data: np.ndarray) -> None:
        """
        Initialize TKM.

        Args:
            data (np.ndarray): Array containing data points. Array should be of size txmxN,
            where t is the number of timesteps, m is the number of dimensions and N is the number of data points.

        Attributes:
            centers (np.ndarray): Array containing cluster centers over all time.
            weights (np.ndarray): Array containing weights determining extent of point membership to a cluster at all times.
            obj_hist (np.ndarray): Value of objective function over all iterations.
            err_hist (np.ndarray): Value of error over all iterations.
            transposed_data (np.ndarray): Transposed data matrix, stored for speed up in computation.
            variable_updates (str): Updates to use based on whether the distance functions in the obj. function are L1, L2, or cos distances.
        """
        self.data = data
        self.transposed_data = None
        self.centers = None
        self.weights = None
        self.obj_hist = None
        self.err_hist = None
        self.variable_updates = None

    def l2_variable_updates(
        self,
        weights: np.ndarray,
        centers: np.ndarray,
        centers_shifted: np.ndarray,
        lam: float = 0.70,
        d_k: float = 1.1,
    ):
        """
        Carry out variable updates for l2 optimization.

        Args:
            weights (np.ndarray): Array containing weights determining extent of point membership to a cluster at all times
            centers (np.ndarray): Array containing cluster centers over all time
            centers_shifted (np.ndarray): Array containing cluster centers over all time, shifted forward by one time step
            lam (float): Paramter controlling extent of penalty constraining cluster centers from moving too far apart
            d_k (float): Parameter controlling speed of converegence

        Returns:
            centers_new (np.ndarray): Updated centers
            weights_new (np.ndarray): Updated weights
            centers_shifted_new (np.ndarray): Updated shifted centers
            err (float): Error between previous centers/weights and current centers/weights
            obj (float): New obj. function value
        """
        _, num_points, _ = weights.shape

        data_norm = np.linalg.norm(self.data, 2, axis=1) ** 2

        centers_new = (self.data @ weights + num_points * lam * centers_shifted) / (
            np.sum(weights, axis=1)[:, np.newaxis, :] + lam * num_points
        )

        centers_norm = np.linalg.norm(centers_new, 2, axis=1) ** 2

        weights_op_constrained = weights - 1 / d_k * (
            data_norm[:, :, np.newaxis]
            - 2 * np.transpose(self.data, axes=[0, 2, 1]) @ centers_new
            + centers_norm[:, np.newaxis, :]
        )

        weights_new = simplex_prox(weights_op_constrained, 1)

        centers_err = np.linalg.norm(centers - centers_new)
        weights_err = np.linalg.norm(weights - weights_new)

        err = d_k * weights_err + centers_err

        centers_shifted_new = np.vstack(
            (centers_new[0, :, :][np.newaxis, :, :], centers_new[:-1, :, :])
        )

        sum_term_1 = np.sum(
            np.linalg.norm(
                self.data - centers_new @ np.transpose(weights_new, axes=[0, 2, 1]),
                2,
                axis=1,
            )
            ** 2
        )

        sum_term_2 = np.sum(
            num_points
            * lam
            * np.linalg.norm(centers_new - centers_shifted_new, 2, axis=1) ** 2
        )

        obj = sum_term_1 + sum_term_2

        return centers_new, weights_new, centers_shifted_new, err, obj

    def cosine_variable_updates(
        self,
        weights: np.ndarray,
        centers: np.ndarray,
        centers_shifted: np.ndarray,
        lam: float = 0.70,
        d_k: float = 1.1,
        gamma: float = 1e-3,
    ):
        """
        Carry out variable updates for cosine optimization.

        Args:
            weights (np.ndarray): Array containing weights determining extent of point membership to a cluster at all times
            centers (np.ndarray): Array containing cluster centers over all time
            centers_shifted (np.ndarray): Array containing cluster centers over all time, shifted forward by one time step
            lam (float): Paramter controlling extent of penalty constraining cluster centers from moving too far apart
            d_k (float): Parameter controlling speed of converegence
            gamma (float): Parameter controlling center update step size

        Returns:
            centers_new (np.ndarray): Updated centers
            weights_new (np.ndarray): Updated weights
            centers_shifted_new (np.ndarray): Updated shifted centers
            err (float): Error between previous centers/weights and current centers/weights
            obj (float): New obj. function value
        """
        ### Data has been transposed when cosine updates are called. ####
        _, _, num_points = self.data.shape

        centers_derivative_term_1 = self.data @ weights
        centers_derivative_term_2 = lam * num_points * centers_shifted
        centers_new = centers - gamma * (
            centers_derivative_term_1 + centers_derivative_term_2
        )

        # Ensure centers are normalized
        centers_new = (
            centers_new / np.linalg.norm(centers_new, 2, axis=1)[:, np.newaxis, :]
        )

        data_center_product = self.transposed_data @ centers_new

        weights_step = weights - 1 / d_k * data_center_product
        weights_new = simplex_prox(weights_step, 1)

        centers_err = np.linalg.norm(centers - centers_new)
        weights_err = np.linalg.norm(weights - weights_new)

        err = d_k * weights_err + 1 / gamma * centers_err

        centers_shifted_new = np.vstack(
            (centers_new[0, :, :][np.newaxis, :, :], centers_new[:-1, :, :])
        )

        sum_term_1 = np.sum(weights_new * data_center_product)

        sum_term_2 = lam * num_points * np.sum(centers_new * centers_shifted_new)

        obj = sum_term_1 + sum_term_2

        return centers_new, weights_new, centers_shifted_new, err, obj

    def l1_variable_updates(
        self,
        weights: np.ndarray,
        centers: np.ndarray,
        centers_shifted: np.ndarray,
        lam: float = 0.70,
        d_k: float = 1.1,
        gamma: float = 1e-3,
    ):
        """
        Carry out variable updates for L1 optimization

         Args:
            weights (np.ndarray): Array containing weights determining extent of point membership to a cluster at all times
            centers (np.ndarray): Array containing cluster centers over all time
            centers_shifted (np.ndarray): Array containing cluster centers over all time, shifted forward by one time step
            lam (float): Paramter controlling extent of penalty constraining cluster centers from moving too far apart
            d_k (float): Parameter controlling speed of converegence
            gamma (float): Parameter controlling center update step size

        Returns:
            centers_new (np.ndarray): Updated centers
            weights_new (np.ndarray): Updated weights
            centers_shifted_new (np.ndarray): Updated shifted centers
            err (float): Error between previous centers/weights and current centers/weights
            obj (float): New obj. function value
        """
        timesteps, num_dimensions, num_clusters = centers.shape
        _, _, num_points = self.data.shape

        term_1 = np.zeros((timesteps, num_dimensions, num_clusters))
        for k in range(num_clusters):
            # weights for a given cluster
            difference = self.data - centers[:, :, k][:, :, np.newaxis]
            sign_difference = np.sign(difference)
            product = weights[:, :, k][:, np.newaxis, :] * sign_difference
            product_sum = np.sum(product, axis=2)
            term_1[:, :, k] = product_sum

        center_difference = centers - centers_shifted
        signed_center_difference = np.sign(center_difference)
        term_2 = lam * num_points * signed_center_difference

        centers_new = centers - gamma * (term_1 + term_2)

        data_center_difference_norm = np.zeros((timesteps, num_points, num_clusters))
        for k in range(num_clusters):
            # weights for a given cluster
            difference = self.data - centers_new[:, :, k][:, :, np.newaxis]
            data_center_difference_norm[:, :, k] = np.linalg.norm(difference, 1, axis=1)

        weights_step = weights - 1 / d_k * data_center_difference_norm
        weights_new = simplex_prox(weights_step, 1)

        centers_err = np.linalg.norm(np.ravel(centers - centers_new), 2)
        weights_err = np.linalg.norm(np.ravel(weights - weights_new), 2)

        err = d_k * weights_err + centers_err

        centers_shifted_new = np.vstack(
            (centers_new[0, :, :][np.newaxis, :, :], centers_new[:-1, :, :])
        )

        sum_term_1 = np.sum(weights_new * data_center_difference_norm)
        sum_term_2 = (
            lam
            * num_points
            * np.sum(np.linalg.norm(centers_new - centers_shifted_new, 1, axis=1))
        )

        obj = sum_term_1 + sum_term_2

        return centers_new, weights_new, centers_shifted_new, err, obj

    def perform_clustering(
        self,
        num_clusters: int,
        tol: float = 1e-4,
        max_iter: int = 100,
        method: str = "L2",
        init_centers: Union[str, np.ndarray] = "kmeans_plus_plus",
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """
        Perform Spatiotemporal k-means algorithm.

        Args:
            num_clusters (int): Number of clusters.
            tol (float): Error tolerance for convergence of algorithm.
            max_iter (int): Max number of iterations for algorithm.
            method (str): Choose distance measure in obj. function. Implemented for 'L2', 'L1', or 'cos'.
            init_centers Union[str, np.ndarray]: Initial centers for algorithm. The default is to choose centers using "kmeans_plus_plus" algorithm. Centers can also be chosen randomly by calling "random" or specified as an array.
            verbose (bool): Whether or not to print statements during convergence.

        Kwargs:
            gamma (float): Parameter controlling center update step size. Only used for L1 or cos optimization.
            lam (float): Parameter controlling strength of constraint that cluster centers should not move over time.
        """
        # Choose variable updates
        if method == "L2":
            self.variable_updates = self.l2_variable_updates
        elif method == "cosine":
            self.data = (
                self.data / np.linalg.norm(self.data, 2, axis=1)[:, np.newaxis, :]
            )
            self.variable_updates = self.cosine_variable_updates
            self.transposed_data = np.transpose(self.data, axes=[0, 2, 1])
        elif method == "L1":
            self.variable_updates = self.l1_variable_updates
        timesteps, _, num_points = self.data.shape

        assert (
            num_clusters <= num_points
        ), "Number of clusters must be less than number of points."

        # Choose initial centers
        if init_centers == "random":
            centers = self.data[:, :, np.random.choice(num_points, num_clusters)]
        elif init_centers == "kmeans_plus_plus":
            _, init_center_ind = kmeans_plusplus(
                self.data[0, :, :].T, n_clusters=num_clusters
            )
            centers = self.data[:, :, init_center_ind]
        else:
            centers = init_centers

        iter_count = 0
        err = tol + 1.0

        obj_hist = []
        err_hist = []

        # Initialize weights randomly
        weights = np.random.rand(timesteps, num_points, num_clusters)

        centers_shifted = np.vstack(
            (centers[0, :, :][np.newaxis, :, :], centers[:-1, :, :])
        )

        while err >= tol:
            (
                centers_new,
                weights_new,
                centers_shifted_new,
                err,
                obj,
            ) = self.variable_updates(
                weights=weights,
                centers=centers,
                centers_shifted=centers_shifted,
                **kwargs,
            )
            np.copyto(centers, centers_new)
            np.copyto(weights, weights_new)
            np.copyto(centers_shifted, centers_shifted_new)

            obj_hist.append(obj)
            err_hist.append(err)

            iter_count += 1

            if verbose is True:
                if iter_count % 100 == 0:
                    print("Iteration", iter_count)

            if iter_count >= max_iter:
                print("Maximum number of iterations")
                self.centers = centers
                self.weights = weights
                self.obj_hist = obj_hist
                self.err_hist = err_hist
                break

        self.centers = centers
        self.weights = weights
        self.obj_hist = obj_hist
        self.err_hist = err_hist
