import pandas as pd
import numpy as np
from distance_functions import s_journey
from tkm.graph_clustering_functions import STGKM, visualize_graph

final_voter_data = pd.read_csv('final_voter_data.csv')
fvd_0 = pd.read_csv('fvd_0')

voter_connectivity = np.load('roll_call_connectivity_2.npy')

distance_matrix = s_journey(voter_connectivity)

np.save('roll_call_distance.npy', distance_matrix)

distance_matrix = np.load('roll_call_distance.npy')

SUBSET = 100
subset_matrix = distance_matrix[:SUBSET]
t,n,_ = distance_matrix.shape
k = 3
stgkm = STGKM(distance_matrix = subset_matrix, penalty = 5, max_drift = 1, center_connectivity = 5, k = k, tie_breaker=False, iterations = 100)
penalized_distance = stgkm.penalize_distance()
# stgkm.first_kmeans(distance_matrix=penalized_distance)
stgkm.run_stgkm(method = 'proxy')
