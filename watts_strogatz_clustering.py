""" Script for running STGKM on Watts Strogratz dynamic graph."""
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from watts_strogatz import WattsStrogatz
from distance_functions import s_journey
from TKM_long_term_clusters import find_final_label_sc
from tkm.graph_clustering_functions import STGKM, visualize_graph

NUMNODES = 8
TIMESTEPS = 100

WS = WattsStrogatz(num_nodes=NUMNODES, num_neighbors=4, probability=0.40)
connectivity_matrix = np.zeros((TIMESTEPS, NUMNODES, NUMNODES))

for time in range(TIMESTEPS):
    connectivity = WS.update()
    connectivity_matrix[time, :, :] = connectivity.toarray()
    # WS.visualize()
    # plt.show()

print("graph created")

#########################

SUBSET = 20
distance_matrix = s_journey(connectivity_matrix=connectivity_matrix)
subset_distance_matrix = distance_matrix[:SUBSET, :,:]
penalty = SUBSET

stgkm = STGKM(distance_matrix=subset_distance_matrix, penalty=penalty, max_drift=1, k=2, tie_breaker = False, 
              iter = 100)

stgkm.run_stgkm(method = 'full')
print(stgkm.ltc)


# visualize_graph(
#     connectivity_matrix=connectivity_matrix[:SUBSET, :, :],
#     labels=stgkm.ltc,
#     centers=stgkm.full_centers,
# )

def illustrate_connectivity(connectivity_matrix, t, ltc):
    timeslice = connectivity_matrix[t]
    num_nodes, _ = timeslice.shape

    #if i and j in cluster 1 - blue 
    #if i and j in cluster  - red
    #if i and j in different clusters- green
    #i and j not connected - 0

    color_matrix = timeslice.copy()
    members = [np.where(ltc == cluster)[0] for cluster in range(2)]

    for i in range(num_nodes):
        for j in range(num_nodes):
            if timeslice[i,j] == 1:
                if (i in members[0]) and (j in members[0]):
                    color_matrix[i,j] = 2
                elif (i in members[1]) and (j in members[1]):
                    color_matrix[i,j] = 3

    plt.imshow(color_matrix)
    plt.show()

    return None

# for t in range(20):
#     illustrate_connectivity(connectivity_matrix, t = t, ltc = stgkm.ltc)




# mems = np.where(stgkm.ltc == 0)[0]
# timeslice = connectivity_matrix[1]
# plt.imshow(timeslice, cmap = 'Greys', interpolation = 'nearest')

##############

# SUBSET = 20
# distance_matrix = s_journey(connectivity_matrix=connectivity_matrix)
# subset_distance_matrix = distance_matrix[:SUBSET, :,:]
# penalty = np.unique(subset_distance_matrix)[-2] + 1

# t,n, _ = subset_distance_matrix.shape
# k = 2

# stgkm = STGKM(distance_matrix=subset_distance_matrix, penalty=penalty, max_drift=1, k=k, tie_breaker = False, 
#               iter = 100)

# full_assignments = np.zeros((t*k, n))
# full_centers = np.zeros((t,k))

# penalized_distance = stgkm.penalize_distance()

# current_members, current_centers = stgkm.first_kmeans(distance_matrix= penalized_distance)

# previous_distance = penalized_distance[0]

# full_assignments[0:k, :] = current_members

# full_centers[0] = current_centers

# for time in range(1, t):
#     current_distance = penalized_distance[time]

#     new_members, new_centers = stgkm.next_assignment(
#         current_centers=current_centers,
#         previous_distance=previous_distance,
#         current_distance=current_distance,
#         )

#     full_centers[time] = new_centers
   
#     full_assignments[time*(k):(time+1)*k,:] = new_members

#     previous_distance = current_distance.copy()
#     current_centers = list(new_centers).copy()

# ltc = find_final_label_sc(weights=self.full_assignments.T, k=self.k)

# print('ltc', ltc)