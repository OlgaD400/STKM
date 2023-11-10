"""Implementation of spatiotemporal k means."""
from typing import Optional, Callable
import numpy as np
from sklearn.cluster import kmeans_plusplus

# from sklearn.cluster import kmeans_plusplus
# from proxlib.operators import proj_csimplex


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


def simplex_prox_2d(z: np.ndarray, a: float) -> np.ndarray:
    """
    Project onto simplex.

    Args:
        z:  Variable to be projected. Assume ``x`` is a matrix,
            each row will be projected onto a simplex.
        a:  Simplex to be projected onto.
    Returns:
        np.maximum(z + lam, 0.0) (np.ndarray): Projected variable.
    """
    u = z.copy()

    if u.ndim == 1:
        u = u[np.newaxis, :]

    u[:, ::-1].sort(axis=1)

    j = np.arange(u.shape[1])
    v = (a - np.cumsum(u, axis=1)) / (j + 1)

    i = np.repeat(j[None, :], u.shape[0], axis=0)
    rho = np.max(i * (u + v > 0), axis=1)

    lam = v[np.arange(u.shape[0]), rho][:, None]

    #     print('\n\n lam', lam, '\n\n')
    return np.maximum(z + lam, 0.0)


class STKM:
    """
    Create a class to perform Time K Means Clustering.

    Minimize the following objective function

    .. math::
        \sum_{j=1}^k \sum_{i = 1}^N \sum_{t = 1}^{t_n} w_{t, :, i} ||x_{t,i} - c_{t,j}||^2 +
        \lambda ||c_t - c_{t+1}||^2 + \gamma ||w_{t,:, i} - w_{t+1, :, i} ||^2

    where

    .. math::
        w_{t,:,i} \in \Delta_1

    so that each row of each time slice sums to 1.

    The centers and the weights are updated iteratively as follows

    .. math::
        c_{t, j}^{l + 1} = \frac{\sum_{i=1}^N w_{t,j,i}^l x_{t, i}^l +
        \lambda N c_{t+1,j}^l}{\sum_{i=1}^N w_{t,j,i}^l + \lambda N}

    .. math::
        w_{t, :, i} ^{l+1} = proj_{\Delta_1}\bigg(w_{t, :, i}^l -
        \frac{1}{d} ||x_{t,i}^l - c_{t,j}^l||^2 -
        2 \gamma (w_{t,:,i}^l - w_{t+1, :,i}^l) \bigg)
    """

    def __init__(self, data: np.ndarray) -> None:
        """
        Initialize TKM.

        Args:
            data (np.ndarray): Array containing data points. Array should be of size mxN,
            where m is the number of dimensions and N is the number of data points.

        Attributes:
            centers (np.ndarray): Array containing calculated cluster centers.
            weights (np.ndarray): Array containing weights determining extent of
              point membership to a cluster.
            obj_hist (np.ndarray): Value of objective function over all iterations.
            err_hist (np.ndarray): Value of error over all iterations.
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
        """Carry out variable updates for l2 optimization."""
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
        """Cosine obj variable updates."""
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
        sum_term_2 = (
            lam
            * num_points
            * np.sum(centers_new @ np.transpose(centers_shifted_new, axes=[0, 2, 1]))
        )

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
        """L1 variable updates."""
        timesteps, num_dimensions, num_clusters = centers.shape
        _, _, num_points = self.data.shape

        term_1 = np.zeros((timesteps, num_dimensions, num_clusters))
        for k in range(num_clusters):
            # weights for a given cluster
            difference = self.data - centers[:, :, k][:, :, np.newaxis]
            sign_difference = np.apply_along_axis(np.sign, axis=1, arr=difference)
            product = weights[:, :, k][:, np.newaxis, :] * sign_difference
            product_sum = np.sum(product, axis=2)
            term_1[:, :, k] = product_sum

        center_difference = centers - centers_shifted
        signed_center_difference = np.apply_along_axis(
            np.sign, axis=1, arr=center_difference
        )
        term_2 = lam * num_points * signed_center_difference

        centers_new = centers - gamma * (term_1 + term_2)

        data_center_difference_norm = np.zeros((timesteps, num_points, num_clusters))
        for k in range(num_clusters):
            # weights for a given cluster
            difference = self.data - centers_new[:, :, k][:, :, np.newaxis]
            data_center_difference_norm[:, :, k] = np.linalg.norm(difference, 1, axis=1)

        weights_step = weights - 1 / d_k * data_center_difference_norm
        weights_new = simplex_prox(weights_step, 1)

        centers_err = np.linalg.norm(centers - centers_new)
        weights_err = np.linalg.norm(weights - weights_new)

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

        return centers_new, weights_new, centers_shifted_new, obj, err

    def perform_clustering(
        self,
        num_clusters: int,
        tol: float = 1e-4,
        max_iter: int = 100,
        method: str = "L2",
        init_centers="kmeans_plus_plus",
        verbose: bool = False,
        **kwargs,
    ) -> None:
        """
        Perform Time k Means algorithm and set values for TKM attributes.

        Sets values for predicted cluster centers, cluster membership weights, outliers,
        objective function values over iterations, and error values over iterations.

        Args:
            num_clusters (int): Number of clusters.
            tol (float): Error tolerance for convergence of algorithm.
            max_iter (int): Max number of iterations for algorithm.
            lam (float): Parameter controlling strength of constraint that
                cluster centers should not move over time.
            init_centers: Initial centers for algorithm.
                Default: centers chosen using kmeans_plus_plus algorithm.
                Can also be chosen randomly. Or specified.
            verbose (bool): Whether or not to print statements during convergence.
        """
        if method == "L2":
            self.variable_updates = self.l2_variable_updates
            transposed_data = None
        elif method == "cosine":
            self.data = (
                self.data / np.linalg.norm(self.data, 2, axis=1)[:, np.newaxis, :]
            )
            self.variable_updates = self.cosine_variable_updates
            self.transposed_data = np.transpose(self.data, axes=[0, 2, 1])
        elif method == "L1":
            self.variable_updates = self.l1_variable_updates
            transposed_data = None
        timesteps, _, num_points = self.data.shape

        assert (
            num_clusters <= num_points
        ), "Number of clusters must be less than number of points."

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

    def perform_clustering_weight_constraint(
        self,
        num_clusters: int,
        tol: float = 1e-4,
        max_iter: int = 100,
        lam: Optional[float] = 0.70,
        gam: Optional[float] = 0.70,
        init_centers: Optional[np.ndarray] = None,
        verbose: bool = False,
        d_k: float = 1.1,
    ) -> None:
        """
        Perform Time k Means algorithm and set values for TKM attributes.

        Sets values for predicted cluster centers, cluster membership weights,
        outliers, objective function values over iterations, and error
        values over iterations.

        Args:
            num_clusters (int): Number of clusters.
            tol (float): Error tolerance for convergence of algorithm.
            max_iter (int): Max number of iterations for algorithm.
            lam (float): Parameter controlling strength of constraint that cluster
                centers should not move over time.
            init_centers (np.ndarray): Initial centers for algorithm.
            verbose (bool): Whether or not to print statements during convergence.
        """
        timesteps, _, num_points = self.data.shape

        if init_centers is None:
            centers = self.data[:, :, np.random.choice(num_points, num_clusters)]
        else:
            centers = init_centers

        # dk = 1.1

        iter_count = 0
        err = tol + 1.0

        obj_hist = []
        err_hist = []

        weights = np.random.rand(timesteps, num_points, num_clusters)

        centers_shifted = np.vstack(
            (centers[0, :, :][np.newaxis, :, :], centers[:-1, :, :])
        )

        weights_shifted = np.vstack(
            (weights[0, :, :][np.newaxis, :, :], weights[:-1, :, :])
        )

        while err >= tol:
            data_norm = np.linalg.norm(self.data, 2, axis=1) ** 2

            centers_new = (self.data @ weights + num_points * lam * centers_shifted) / (
                np.sum(weights, axis=1)[:, np.newaxis, :] + lam * num_points
            )

            centers_norm = np.linalg.norm(centers_new, 2, axis=1) ** 2

            weights_difference = (weights - weights_shifted) / np.abs(
                weights - weights_shifted
            )

            weights_op_constrained = (
                weights
                - 1
                / d_k
                * (
                    data_norm[:, :, np.newaxis]
                    - 2 * np.transpose(self.data, axes=[0, 2, 1]) @ centers_new
                    + centers_norm[:, np.newaxis, :]
                )
                - gam / d_k * weights_difference
            )

            weights_new = simplex_prox(weights_op_constrained, 1)

            centers_err = np.linalg.norm(centers - centers_new)
            weights_err = np.linalg.norm(weights - weights_new)

            np.copyto(centers, centers_new)
            np.copyto(weights, weights_new)
            err = d_k * weights_err + centers_err

            centers_shifted = np.vstack(
                (centers_new[0, :, :][np.newaxis, :, :], centers_new[:-1, :, :])
            )

            weights_shifted = np.vstack(
                (weights[0, :, :][np.newaxis, :, :], weights[:-1, :, :])
            )

            sum_term_1 = np.sum(
                np.linalg.norm(
                    self.data - centers @ np.transpose(weights, axes=[0, 2, 1]),
                    2,
                    axis=1,
                )
                ** 2
            )

            sum_term_2 = np.sum(
                num_points
                * lam
                * np.linalg.norm(centers_new - centers_shifted, 2, axis=1) ** 2
            )

            sum_term_3 = gam * np.sum(np.abs(weights - weights_shifted))

            obj = sum_term_1 + sum_term_2 + sum_term_3

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

            # dk += .01

            # dk = dk/(1 + .9*iter_count)

        self.centers = centers
        self.weights = weights
        self.obj_hist = obj_hist
        self.err_hist = err_hist

    def perform_clustering_avg_centers(
        self,
        num_clusters: int,
        tol: float = 1e-4,
        max_iter: int = 100,
        lam: Optional[float] = 0.70,
        init_centers: Optional[np.ndarray] = None,
        verbose: bool = False,
    ) -> None:
        """
        Perform Time k Means algorithm and set values for TKM attributes.

        Sets values for predicted cluster centers, cluster membership weights,
        outliers, objective function values over iterations, and error values
        over iterations.

        Args:
            num_clusters (int): Number of clusters.
            tol (float): Error tolerance for convergence of algorithm.
            max_iter (int): Max number of iterations for algorithm.
            lam (float): Parameter controlling strength of constraint that cluster
                centers should not move over time.
            init_centers (np.ndarray): Initial centers for algorithm.
            verbose (bool): Whether or not to print statements during convergence.
        """
        timesteps, _, num_points = self.data.shape

        if init_centers is None:
            centers = self.data[:, :, np.random.choice(num_points, num_clusters)]
        else:
            centers = init_centers

        d_k = 1.1

        iter_count = 0
        err = tol + 1.0

        obj_hist = []
        err_hist = []

        weights = np.transpose(
            np.dstack([np.random.rand(num_points, num_clusters)] * timesteps),
            axes=[2, 0, 1],
        )
        centers_average = (
            np.cumsum(centers, axis=0)
            / np.arange(1, timesteps + 1)[:, np.newaxis, np.newaxis]
        )

        while err >= tol:
            data_norm = np.linalg.norm(self.data, 2, axis=1) ** 2

            centers_new = (self.data @ weights + num_points * lam * centers_average) / (
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

            np.copyto(centers, centers_new)
            np.copyto(weights, weights_new)
            err = d_k * weights_err + centers_err

            centers_average = (
                np.cumsum(centers, axis=0)
                / np.arange(1, timesteps + 1)[:, np.newaxis, np.newaxis]
            )

            sum_term_1 = np.sum(
                np.linalg.norm(
                    self.data - centers @ np.transpose(weights, axes=[0, 2, 1]),
                    2,
                    axis=1,
                )
                ** 2
            )

            sum_term_2 = np.sum(
                num_points
                * lam
                * np.linalg.norm(centers_new - centers_average, 2, axis=1) ** 2
            )

            obj = sum_term_1 + sum_term_2

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

    def perform_clustering_log_c(
        self,
        num_clusters: int,
        tol: float = 1e-6,
        max_iter: int = 100,
        lam: Optional[float] = 0.70,
        init_centers: Optional[np.ndarray] = None,
        nu: Optional[float] = 1.1,
    ) -> None:
        """
        Perform Time k Means algorithm for an objective function that is robust
        to outliers and noise.

        Sets values for predicted cluster centers, cluster membership weights,
        outliers, objective function values over iterations, and error values
        over iterations.

        Args:
            num_clusters (int): Number of clusters.
            tol (float): Error tolerance for convergence of algorithm.
            max_iter (int): Max number of iterations for algorithm.
            lam (float): Parameter controlling strength of constraint that cluster
                centers should not move over time.
            init_centers (np.ndarray): Initial centers for algorithm.
        """
        timesteps, dimension, num_points = self.data.shape

        if init_centers is None:
            centers = self.data[:, :, np.random.choice(num_points, num_clusters)]
        else:
            centers = init_centers

        d_k = 1.1
        e_k = 1.1 * (nu + 1) * num_points

        iter_count = 0
        err = tol + 1.0

        obj_hist = []
        err_hist = []

        weights = np.random.rand(timesteps, num_points, num_clusters)

        centers_shifted = np.vstack(
            (centers[0, :, :][np.newaxis, :, :], centers[:-1, :, :])
        )

        while err >= tol:
            data_norm = np.linalg.norm(self.data, 2, axis=1) ** 2
            centers_norm = np.linalg.norm(centers, 2, axis=1) ** 2

            denominator = (
                nu
                + data_norm[:, :, np.newaxis]
                - 2 * np.transpose(self.data, axes=[0, 2, 1]) @ centers
                + centers_norm[:, np.newaxis, :]
            )

            centers_sum = np.zeros((timesteps, dimension, num_clusters))

            for i in range(num_points):
                numerator = weights[:, i, :][:, np.newaxis, :] * (
                    self.data[:, :, i][:, :, np.newaxis] - centers
                )
                centers_sum += numerator / (denominator[:, i, :][:, np.newaxis, :])

            centers_new = (
                (nu + 1) * centers_sum + e_k * centers + 2 * lam * centers_shifted
            ) / (2 * lam + e_k)

            centers_norm = np.linalg.norm(centers_new, 2, axis=1) ** 2

            weights_op_constrained = weights - (nu + 1) / (2 * d_k) * np.log(
                data_norm[:, :, np.newaxis]
                - 2 * np.transpose(self.data, axes=[0, 2, 1]) @ centers_new
                + centers_norm[:, np.newaxis, :]
                + nu
            )

            weights_new = simplex_prox(weights_op_constrained, 1)

            centers_err = np.linalg.norm(centers - centers_new)
            weights_err = np.linalg.norm(weights - weights_new)

            np.copyto(centers, centers_new)
            np.copyto(weights, weights_new)

            err = weights_err * d_k + centers_err

            centers_shifted = np.vstack(
                (centers_new[0, :, :][np.newaxis, :, :], centers_new[:-1, :, :])
            )

            sum_term_1 = np.sum(
                weights_new
                * (nu + 1)
                / 2
                * np.log(
                    data_norm[:, :, np.newaxis]
                    - 2 * np.transpose(self.data, axes=[0, 2, 1]) @ centers_new
                    + centers_norm[:, np.newaxis, :]
                    + nu
                )
            )

            sum_term_2 = np.sum(
                lam * np.linalg.norm(centers_new - centers_shifted, 2, axis=1) ** 2
            )

            obj = sum_term_1 + sum_term_2

            obj_hist.append(obj)
            err_hist.append(err)

            iter_count += 1

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
