""" Roll Call Data Clustering Script"""

import pandas as pd
import numpy as np
from stgkm.distance_functions import s_journey
from stgkm.graph_clustering_functions import STGKM

final_voter_data = pd.read_csv('final_voter_data.csv')
fvd_0 = pd.read_csv('fvd_0')

voter_connectivity = np.load('roll_call_connectivity_2.npy')

distance_matrix = s_journey(voter_connectivity)

np.save('roll_call_distance.npy', distance_matrix)

distance_matrix = np.load('roll_call_distance.npy')

#Run STGkM on a subset of 100 roll call votes
SUBSET = 100
subset_matrix = distance_matrix[:SUBSET]
TIME,NUM_VERTICES,_ = distance_matrix.shape
NUM_CLUSTERS = 3
stgkm = STGKM(distance_matrix = subset_matrix, penalty = 5, max_drift = 1,
              drift_time_window = 5, num_clusters = NUM_CLUSTERS,
              tie_breaker=False, iterations = 100)
penalized_distance = stgkm.penalize_distance()
stgkm.run_stgkm(method = 'proxy')
