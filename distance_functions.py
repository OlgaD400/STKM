import numpy as np

def temporal_graph_distance(connectivity_matrix):
    distance_matrix = np.copy(connectivity_matrix)
    t,n,n = distance_matrix.shape

    #Replace all off-diagonal zeros with infinity
    diagonal_mask = np.array([np.eye(n, dtype = bool)]*t)
    distance_matrix = np.where((distance_matrix == 0)&(~diagonal_mask), np.inf, distance_matrix)

    #iterate backwards through each time slice
    for time in np.arange(t-1)[::-1]:
        slice = distance_matrix[time, :,:]
        next_slice = distance_matrix[time+1,:,:]

        #iterate through each row
        for row in range(n):
            connections = np.where(slice[row] == 1)[0]
            ind_to_update = np.where(slice[row]>1)[0]

            #If there are no connections, there's nothing to update
            #If there are no np.infs, there's nothing to update
            if (len(connections)>0) & (len(ind_to_update)>0):
                if len(connections)>1:
                    connections_distance = np.min(next_slice[connections,ind_to_update], axis = 0) + 1
                else:
                    connections_distance = next_slice[connections,ind_to_update] + 1
                slice[row, ind_to_update] = np.minimum(slice[row, ind_to_update], connections_distance)
            else: 
                continue
    
    #Replace any nonzero entries on diagonals with zero
    distance_matrix = np.where(diagonal_mask, 0, distance_matrix)
    return distance_matrix