import numpy as np
from distance_functions import temporal_graph_distance
from TKM_long_term_clusters import find_final_label_sc
import itertools

def test_temporal_graph_distance():
        """
        Test temporal graph distance
        """

        connectivity_matrix = np.array([[[0,1,0,0], [1,0,1,0], [0,1,0,1], [0,0,1,0]], 
                                        [[0,1,0,0], [1,0,0,1], [0,0,0,1], [0,1,1,0]],
                                        [[0,0,0,1], [0,0,1,1], [0,1,0,0], [1,1,0,0]],
                                        [[0,1,1,0], [1,0,1,0], [1,1,0,0], [0,0,0,0]]])

        # t,n,n = connectivity_matrix.shape
        # #Ensure test cases are symmetric
        # for i in range(t):
        #     assert np.all(connectivity_matrix[i,:,:] == connectivity_matrix[i,:,:].T)
    
        distance_matrix = temporal_graph_distance(connectivity_matrix)
        assert np.all(distance_matrix == np.array([[[0,1,4,2], [1,0,1,2], [2,1,0,1], [3,3,1,0]], 
                                            [[0,1,2,2], [1,0,3,1], [2,2,0,1], [3,1,1,0]],
                                             [[0, np.inf, np.inf, 1], [2,0,1,1], [2,1,0,np.inf], [1,1,2,0]], 
                                             [[0, 1, 1, np.inf], [1, 0, 1, np.inf], [1,1,0, np.inf], [np.inf, np.inf, np.inf, 0]]])) 
        
        connectivity_matrix = np.array([[[0,1,0,0], [1,0,1,1], [0,1,1,1], [0,1,1,0]],
                                       [[1,0,1,0], [0,0,1,1], [1,1,0,0], [0,1,0,1]],
                                       [[0,1,0,0], [1,0,0,0], [0,0,0,1], [0,0,1,0]]])
        distance_matrix = temporal_graph_distance(connectivity_matrix)
        assert np.all(distance_matrix == np.array([[[0,1,2,2], [1,0,1,1], [2,1,0,1], [2,1,1,0]],
            [[0,2,1,2], [np.inf,0,1,1], [1,1,0,np.inf], [2,1,2,0]],
            [[0, 1, np.inf, np.inf], [1, 0, np.inf, np.inf], [np.inf, np.inf, 0, 1], [np.inf, np.inf, 1, 0]]]))

test_temporal_graph_distance()

two_cluster_connectivity_matrix = np.array([[[0,0,1,0,0,0], [0,0,1,1,0,0], [1,1,0,0,0,0], [0,1,0,0,1,1], [0,0,0,1,0,0], [0,0,0,1,0,0]],
                                    [[0,1,0,0,0,0], [1,1,1,0,1,0], [0,1,0,0,0,0], [0,0,0,0,0,1], [0,1,0,0,0,1], [0,0,0,1,1,1]],
                                    [[1,0,0,0,0,0], [0,0,1,0,0,0], [0,1,1,1,0,0], [0,0,1,0,1,1], [0,0,0,1,0,1], [0,0,0,1,1,0]],
                                    [[0,1,1,0,0,0], [1,0,1,0,0,0], [1,1,0,0,0,0], [0,0,0,0,0,1], [0,0,0,0,1,1], [0,0,0,1,1,0]]])


distance_matrix = temporal_graph_distance(two_cluster_connectivity_matrix)
# print(distance_matrix)

def calculate_center_vertices(distance_matrix, membership, k):
        """
        membership: txn matrix storing a point's cluster membership 
        at every time step
        Update: There should be cluster centers/members at every timestep
        assignment matrix can track which vertex belongs to each cluster at every time step
        
        Output:
        t x k matrix containing k clusters at every timestep
        """
        
        t,_ = membership.shape


        centers = np.zeros((t,k))

        for time in range(t):
            for cluster in range(k):
                  members = np.where(membership[time] == cluster)[0]
                  member_distances = np.sum(distance_matrix[time, members, :][:, members], axis = 0)
                  centers_t_k = members[np.argmin(member_distances)]
                  centers[time,cluster] = centers_t_k
                
        
        return centers.astype(int)

def calculate_center_vertices_2(distance_matrix, membership, previous_centers):
        """
        membership: txn matrix storing a point's cluster membership 
        at every time step
        Update: There should be cluster centers/members at every timestep
        assignment matrix can track which vertex belongs to each cluster at every time step
        
        Output:
        t x k matrix containing k clusters at every timestep
        """
        
        t, k = previous_centers.shape

        centers = np.zeros((t,k))

        for time in range(t):
            #rows associated with previous cluster centers
            for cluster in range(k):
                  #center from previous iteration
                  previous_center = previous_centers[time,cluster]
                  #members of cluster based on centers from previous iteration
                  members = np.where(membership[time] == cluster)[0]
                  #distance between all points in cluster
                  member_distances = np.sum(distance_matrix[time, members, :][:, members], axis = 0)

                  #temporal distance from previous center to all other centers
                  center_distance = distance_matrix[time, previous_center,:][members]
                #   print('distances', member_distances, center_distance)
                #   print('total distance', member_distances + center_distance)
                 
                  centers_t_k = members[np.argmin(member_distances+center_distance)]
                  centers[time,cluster] = centers_t_k
                
        
        return centers.astype(int)

def assign_vertices(distance_matrix: np.ndarray, center_vertices: np.ndarray):
    """
    Assign each point to its closest cluster center 
    
    distance_matrix: np.ndarray t x n x n matrix 
    center_vertices: np.ndarray t x k matrix containing k clusters at every time step
    """
    t,n,n = distance_matrix.shape
    t,k = center_vertices.shape

    times = np.repeat(np.arange(t),k)
    center_distances = distance_matrix[times, np.reshape(center_vertices, ((1,-1)))].reshape((t,k,n))

    membership = np.argmin(center_distances, axis =1)

    return membership


# center_vertices = np.array([[3,4], [0,4], [1, 4], [0,4]])
# center_vertices = np.array([[3,4], [3,4], [3, 4], [3,4]])

# for i in range(10):
#     membership = assign_vertices(distance_matrix=distance_matrix, center_vertices=center_vertices)
#     print('\n\n', membership, '\n\n')

#     center_vertices = calculate_center_vertices(distance_matrix=distance_matrix, membership = membership, k=2)

#     print(center_vertices, '\n\n')
# ltc = find_final_label_sc(weights = membership.T, k = 2)
# print(ltc)

# previous_center_vertices = np.array([[3,4], [0,4], [1, 4], [0,4]])
# previous_center_vertices = np.array([[3,4], [3,4], [3, 4], [3,4]])
# previous_center_vertices = np.array([[0,1], [0,1], [0,1], [0,1]])

# center_vertices = np.copy(previous_center_vertices)

# for i in range(10):
#     membership = assign_vertices(distance_matrix=distance_matrix, center_vertices=center_vertices)
#     print('\n\n', membership, '\n\n')

#     center_vertices = calculate_center_vertices_2(distance_matrix=distance_matrix, membership = membership, previous_centers = previous_center_vertices)
#     previous_center_vertices = np.copy(center_vertices)
    
#     print('vertices', center_vertices, '\n\n')
# ltc = find_final_label_sc(weights = membership.T, k = 2)
# print('ltc', ltc)


#penalty for not being connected
#maximum drift between cluster centers

def penalize_distance(distance_matrix, penalty):
    penalized_distance = np.where(distance_matrix == np.inf, penalty, distance_matrix)
    return penalized_distance

def first_kmeans(init_matrix, k):
    init_centers = np.argsort(np.sum(init_matrix, axis =1))[:k]

    centers = init_centers.copy()

    for iter in range(10):
        #assign each point to its closest cluster center
        center_distances = init_matrix[centers, :]
        membership = np.argmin(center_distances, axis =0)

        #reassign centers based on new membership 
        for cluster in range(k):
                members = np.where(membership == cluster)[0]

                member_distances = np.sum(init_matrix[members,:][:, members], axis = 0)
                
                center_k = members[np.argmin(member_distances)]

                centers[cluster] = center_k     
    
    return membership, centers


def next_assignment(current_centers, previous_distance, current_distance, max_drift):
      #Find all vertices that are within max_drift distance of each current center
      k = len(current_centers)
     
      center_connections = [np.where(previous_distance[center,:] <= max_drift)[0] for center in current_centers]
    
      min_sum = np.sum(current_distance[current_centers,:])

      for center_combination in itertools.product(*center_connections):
        #all chosen centers are unique
        if len(set(center_combination)) == k:
            #This will iterate through every possible subset of centers
        
            #Assign each point to its closest cluster center 
            center_distances = current_distance[center_combination, :]
            membership = np.argmin(center_distances, axis = 0)
            
            #get total sum of distances from vertex in cluster
            cluster_members = [np.where(membership == cluster)[0] for cluster in range(k)]
            
            total_sum = np.sum([np.sum(current_distance[center, mem
                                                        bers]) for center, members in zip(center_combination,cluster_members)])
            #Return centers with smallest distances from their members
            if total_sum < min_sum:
                final_centers = center_combination
                final_members = membership
        
      return final_members, final_centers

t,n,_ = distance_matrix.shape
penalized_distance = penalize_distance(distance_matrix = distance_matrix, penalty = 3)
previous_members, current_centers = first_kmeans(init_matrix=penalized_distance[0], k=2)

previous_distance = penalized_distance[0]

print('chosen centers', current_centers, '\n\n')

total_membership = np.zeros((t, n))
total_membership[0] = previous_members

for time in range(1,t):
     current_distance = penalized_distance[time]
     new_members, new_centers = next_assignment(current_centers= current_centers, previous_distance = previous_distance, 
                     current_distance = current_distance, max_drift = 1)
     
     previous_distance = current_distance.copy()
     current_centers = list(new_centers).copy()

     total_membership[time] = new_members
     print(new_members)
     print(current_centers)

ltc = find_final_label_sc(weights = total_membership.T, k = 2)
print('ltc', ltc)
      