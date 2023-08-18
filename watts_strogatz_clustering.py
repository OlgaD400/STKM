from watts_strogatz import WattsStrogatz
from distance_functions import temporal_graph_distance
from TKM_long_term_clusters import find_final_label_sc
import numpy as np
import matplotlib.pyplot as plt
from graph_clustering_functions import STGKM
import networkx as nx
from graph_clustering_functions import visualize_graph

n = 8
t = 20

WS = WattsStrogatz(n = n, q = 3, probability = .40)
connectivity_matrix = np.zeros((t, n, n))

for time in range(t):
    connectivity = WS.update()
    connectivity_matrix[time, :, :] = connectivity.toarray()
    # WS.visualize()
    # plt.show()

print('graph created')

#########################
distance_matrix = temporal_graph_distance(connectivity_matrix=connectivity_matrix)
penalty = np.unique(distance_matrix)[-2] + 1

stgkm = STGKM (distance_matrix=distance_matrix, penalty = penalty, max_drift = 1, k = 3)

stgkm.run_stgkm()
print(stgkm.ltc)


visualize_graph(connectivity_matrix=connectivity_matrix, labels = stgkm.full_assignments, centers = stgkm.full_centers)


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
#      new_members, new_centers = stgkm.next_assignment(current_centers= current_centers, previous_distance = previous_distance, 
#                      current_distance = current_distance)
     
#      previous_distance = current_distance.copy()
#      current_centers = list(new_centers).copy()

#      total_membership[time] = new_members
#      print(new_members)
#      print(current_centers)

# # print(current_centers)
# ltc = find_final_label_sc(weights = total_membership.T, k = 2)
# print('ltc', ltc)



