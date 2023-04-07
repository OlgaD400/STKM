import numpy as np
import matplotlib.pyplot as plt
from TKM import TKM 
from TKM_long_term_clusters import find_final_label_sc


def correlated_random_walk(intra_cluster_correlations: np.ndarray, 
                           cluster_populations: np.ndarray, 
                           timesteps: int,
                           m: int =2) -> np.ndarray:
    
    """
    Generate the coordinates of a correlated random walk where 
    \math::
        X_i^t = X_i^{t-1} + Y_i^{t-1}
    and
    \math::
        Y_i^t \sim \mathcal{N}\left(\bm{0}, \mathbb{I}_{d}\right)
        \Cora{Y_{i}^{s}}{Y_{j}^{t}} =
        \begin{cases}
            p & a(i) = a(j), s = t \\
            0 & \text{otherwise}
        \end{cases}

    Args:
        intra_cluster_correlations (np.ndarray): Array containing within-cluster point correlations
        cluster_populations (np.ndarray): Array containing populations of points in each cluster
        timesteps (int): Number of timesteps to extend the simulation 

    Returns:
        coordinates (np.ndarray): Array containing the coordinates of all points in the simulation over timesteps
    """
    
    num_populations = len(cluster_populations)

    diagonal_blocks = [np.ones((cluster_populations[i], cluster_populations[i]))*intra_cluster_correlations[i] +
                               np.eye(cluster_populations[i])*(1-intra_cluster_correlations[i])
                 for i in range(num_populations)]
    
    
    correlation_matrix = np.zeros((sum(cluster_populations), sum(cluster_populations)))
    prev_index = 0 
    for index, block in enumerate(diagonal_blocks):
        next_index = prev_index + cluster_populations[index]
        correlation_matrix[prev_index:next_index, prev_index:next_index] = block
        prev_index = next_index
    rng = np.random.default_rng()
    y = rng.multivariate_normal(np.zeros(sum(cluster_populations)), correlation_matrix, size=(timesteps,m))
    
    coordinates = np.vstack((np.zeros((1, m, sum(cluster_populations))),np.cumsum(y, axis = 0)))
    return coordinates

T = 100
cluster_populations = [2,2]
intra_cluster_correlations = [.9,.7,]
coordinates = correlated_random_walk(intra_cluster_correlations = intra_cluster_correlations, 
                                     cluster_populations=cluster_populations, timesteps = T,
                                     m = 1)
colors = ['r', 'b', 'g']

print(coordinates.shape)

##### Figure for 2D
# plt.figure()
# prev_index = 0
# for index, val in enumerate(cluster_populations):
#     next_index = prev_index + val
#     for i in range(prev_index, next_index):
#         plt.scatter(coordinates[0,0, i], coordinates[0,1,i], c = 'g')
#         plt.scatter(coordinates[-1,0, i], coordinates[-1,1,i], c = 'k')
#         plt.plot(coordinates[:,0, i], coordinates[:,1,i], c = colors[index])
#     prev_index = next_index


###### Figure for 1D
plt.figure()
prev_index = 0
for index, val in enumerate(cluster_populations):
    next_index = prev_index + val
    for i in range(prev_index, next_index):
        plt.scatter(0, coordinates[0,0,i], c = 'g')
        plt.scatter(T+1, coordinates[-1,0,i], c = 'k')
        plt.plot(np.arange(T+1), coordinates[:,0,i], c = colors[index])
    prev_index = next_index

tkm = TKM(coordinates)
tkm.perform_clustering(k=3, lam=.80, max_iter=500)

ltc = find_final_label_sc(tkm.weights, k = len(cluster_populations))
print(ltc)
