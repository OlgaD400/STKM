"""Temporal Graph Clustering Script"""
import random
import networkx as nx
import numpy as np
from distance_functions import s_journey
from tkm.graph_clustering_functions import STGKM, visualize_graph


def three_cluster_connectivity(pop_size:int, num_changes:int) -> np.ndarray:
    """
    Create three cluster connectivity matrix.

    Intracluster nodes are fully connected to begin. At each time step, up to 
    num_changes edges are dropped within clusters and added between clusters.

    Args:
        pop_size (int): Size of population of each of three clusters
        changes (int): Max number of edges to drop or add at each time step
    Returns:
        cluster_connectivity_matrix (np.ndarray) Three cluster connectivity matrix.
    """
    cluster = np.ones((pop_size,pop_size))
    zeros = np.zeros((pop_size,pop_size))
    three_clusters = np.block([[cluster, zeros, zeros],
                               [zeros, cluster, zeros],
                               [zeros, zeros, cluster]])
    cluster_connectivity_matrix = np.repeat([three_clusters], 20, axis = 0)

    for time_slice in range(20):
        connectivity_slice = cluster_connectivity_matrix[time_slice]
        ones_x, ones_y = np.where(connectivity_slice == 1)
        zeros_x, zeros_y = np.where(connectivity_slice == 0)

        num_ones = np.arange(len(ones_x))
        num_zeros = np.arange(len(zeros_x))

        indices_to_delete = np.random.choice(num_ones, random.randint(0,num_changes))
        indices_to_add = np.random.choice(num_zeros, random.randint(0,num_changes))

        for indices_y in ones_y[indices_to_delete]:
            for indices_x in ones_x[indices_to_delete]:
                connectivity_slice[indices_x, indices_y] = 0
        for indices_y in zeros_y[indices_to_add]:
            for indices_x in zeros_x[indices_to_add]:
                connectivity_slice[indices_x, indices_y] = 1

    return cluster_connectivity_matrix

two_cluster_connectivity_matrix = np.array(
    [
        [
            [0, 0, 1, 0, 0, 0],
            [0, 0, 1, 1, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 1, 0, 0, 1, 1],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0, 0],
        ],
        [
            [0, 1, 0, 0, 0, 0],
            [1, 1, 1, 0, 1, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 1, 0, 0, 0, 1],
            [0, 0, 0, 1, 1, 1],
        ],
        [
            [1, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 1, 1, 1, 0, 0],
            [0, 0, 1, 0, 1, 1],
            [0, 0, 0, 1, 0, 1],
            [0, 0, 0, 1, 1, 0],
        ],
        [
            [0, 1, 1, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],
            [1, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 1, 1, 0],
        ],
    ]
)                              
#three_cluster_connectivity_matrix = three_cluster_connectivity(pop_size = 10,
#num_changes = 30)

g = nx.Graph(two_cluster_connectivity_matrix[0])
pos = nx.spring_layout(g)

visualize_graph(connectivity_matrix= two_cluster_connectivity_matrix)
distance_matrix = s_journey(two_cluster_connectivity_matrix)

stgkm = STGKM(distance_matrix = distance_matrix[:10], penalty = 8, max_drift = 1,
              center_connectivity = 1, k = 3, tie_breaker=False,
              iterations = 100)
stgkm.run_stgkm(method = 'full')

visualize_graph(connectivity_matrix=two_cluster_connectivity_matrix, labels = stgkm.ltc,
                centers = stgkm.full_centers)
print(stgkm.ltc)
