""" Script for running STGKM on Watts Strogratz dynamic graph."""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from watts_strogatz import WattsStrogatz
from distance_functions import temporal_graph_distance
from TKM_long_term_clusters import find_final_label_sc
from tkm.graph_clustering_functions import STGKM, visualize_graph

NUMNODES = 8
TIMESTEPS = 100

WS = WattsStrogatz(num_nodes=NUMNODES, num_neighbors=3, probability=0.40)
connectivity_matrix = np.zeros((TIMESTEPS, NUMNODES, NUMNODES))

for time in range(TIMESTEPS):
    connectivity = WS.update()
    connectivity_matrix[time, :, :] = connectivity.toarray()
    # WS.visualize()
    # plt.show()

print("graph created")

#########################
distance_matrix = temporal_graph_distance(connectivity_matrix=connectivity_matrix)
penalty = np.unique(distance_matrix)[-2] + 1

stgkm = STGKM(distance_matrix=distance_matrix, penalty=penalty, max_drift=1, k=2)

# stgkm.run_stgkm_proxy()
stgkm.run_stgkm()
print(stgkm.ltc)


visualize_graph(
    connectivity_matrix=connectivity_matrix,
    labels=stgkm.full_assignments,
    centers=stgkm.full_centers,
)


##############

# penalized_distance = stgkm.penalize_distance()
# previous_members, current_centers = stgkm.first_kmeans()

# previous_distance = penalized_distance[0]

# total_membership = np.zeros((t, n))
# total_membership[0] = previous_members

# print('starting centers', current_centers)
# print('starting_membership', previous_members)

# for time in range(1,t):
#      current_distance = penalized_distance[time]
#      new_members, new_centers = stgkm.next_assignment(current_centers= current_centers,
#                                                      previous_distance = previous_distance,
#                                                      current_distance = current_distance)

#      previous_distance = current_distance.copy()
#      current_centers = list(new_centers).copy()

#      total_membership[time] = new_members
#      print(new_members)
#      print(current_centers)

# print(current_centers)
# ltc = find_final_label_sc(weights = total_membership.T, k = 2)
# print('ltc', ltc)
