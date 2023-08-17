import numpy as np
from TKM_long_term_clusters import find_final_label_sc
import itertools


class STGKM():
    def __init__(self, distance_matrix, penalty, max_drift, k):
          self.penalty = penalty
          self.max_drift = max_drift
          self.k = k
          self.distance_matrix = distance_matrix

          t,n,_ = self.distance_matrix.shape

          self.full_centers = np.zeros((t, k))
          self.full_assignments = np.zeros((t, n))
          self.ltc = np.zeros(n)

    def penalize_distance(self):
        penalized_distance = np.where(self.distance_matrix == np.inf, self.penalty, self.distance_matrix)
        return penalized_distance

    def first_kmeans(self):
        init_centers = np.argsort(np.sum(np.sum(self.distance_matrix, axis =2), axis = 0))[:self.k]
        init_matrix = self.distance_matrix[0]

        centers = init_centers.copy()

        for iter in range(10):
            #assign each point to its closest cluster center
            center_distances = init_matrix[centers, :]
            membership = np.argmin(center_distances, axis =0)

            #reassign centers based on new membership 
            for cluster in range(self.k):
                    members = np.where(membership == cluster)[0]

                    member_distances = np.sum(init_matrix[members,:][:, members], axis = 0)
                    
                    if len(member_distances) > 0:
                        #points were assigned to that cluster
                        center_k = members[np.argmin(member_distances)]
                    else:
                        center_k = centers[cluster]

                    centers[cluster] = center_k     
        
        return membership, centers


    def next_assignment(self, current_centers, previous_distance, current_distance):
        #Find all vertices that are within max_drift distance of each current center        
        center_connections = [np.where(previous_distance[center,:] <= self.max_drift)[0] for center in current_centers]
        
        #Preference to keep cluster centers the same
        center_distances = current_distance[current_centers, :]
        current_membership = np.argmin(center_distances, axis = 0)        
        current_members = [np.where(current_membership == cluster)[0] for cluster in range(self.k)]
        min_sum = np.sum([np.sum(current_distance[center, members]) for center, members in zip(current_centers,current_members)])
        final_members = current_membership
        final_centers = current_centers

        for center_combination in itertools.product(*center_connections):
            #all chosen centers are unique
            if len(set(center_combination)) == self.k:
                #This will iterate through every possible subset of centers
            
                #Assign each point to its closest cluster center 
                center_distances = current_distance[center_combination, :]
                membership = np.argmin(center_distances, axis = 0)
                
                #get total sum of distances from vertex in cluster
                cluster_members = [np.where(membership == cluster)[0] for cluster in range(self.k)]
                
                total_sum = np.sum([np.sum(current_distance[center, members]) for center, members in zip(center_combination,cluster_members)])
                #Return centers with smallest distances from their members
                
                if total_sum < min_sum:
                    final_centers = center_combination
                    final_members = membership
                    min_sum = total_sum
            
        return final_members, final_centers
    
    def run_stgkm(self):
        t,n,_ = self.distance_matrix.shape
        penalized_distance = self.penalize_distance()
        current_members, current_centers = self.first_kmeans()

        previous_distance = penalized_distance[0]

        self.full_assignments[0] = current_members
        self.full_centers[0] = current_centers

        for time in range(1,t):
            current_distance = penalized_distance[time]
            new_members, new_centers = self.next_assignment(current_centers= current_centers, current_membership = current_members, previous_distance = previous_distance, 
                            current_distance = current_distance)
            
            self.full_centers[time] = new_centers
            self.full_assignments[time] = new_members

            previous_distance = current_distance.copy()
            current_centers = list(new_centers).copy()    

        self.ltc = find_final_label_sc(weights = self.full_assignments.T, k = self.k)
        
        return None