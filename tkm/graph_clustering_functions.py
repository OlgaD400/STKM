import numpy as np
from TKM_long_term_clusters import find_final_label_sc
from typing import Literal
import itertools
import networkx as nx
import matplotlib.pyplot as plt
import random

class STGKM:
    def __init__(self, distance_matrix, penalty, max_drift, k, iterations=10, center_connectivity=1, tie_breaker: bool = False):
        self.penalty = penalty
        self.max_drift = max_drift
        self.k = k
        self.distance_matrix = distance_matrix
        self.center_connectivity = center_connectivity

        self.t, self.n, _ = self.distance_matrix.shape

        self.full_centers = np.zeros((self.t, self.k))
        self.tie_breaker = tie_breaker

        if self.tie_breaker is True:
            self.full_assignments = np.zeros((self.t, self.n))
        else:
            self.full_assignments = np.zeros((self.t*self.k, self.n))
        self.ltc = np.zeros(self.n)
        self.iter = iterations

    def penalize_distance(self):
        penalized_distance = np.where(
            self.distance_matrix == np.inf, self.penalty, self.distance_matrix
        )
        return penalized_distance
    
    def assign_points(self, distance_matrix: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """
        Assign each point to its closest cluster center.

        If self.tie_breaker is true, each point can only be assinged to one cluster. Else, a point can
        be assigned to multiple clusters

        Args:
            distance_matrix (np.ndarray): Distance between all pairs of points
            centers (np.ndarray): Indices of cluster centers
        
        Returns:
            membership (np.ndarray) Array containing binary assignment matrix with a 1 at index i,j
                if point i belongs to cluster j


        """
        center_distances = distance_matrix[centers, :]
        min_center_distances = np.min(center_distances, axis = 0)
        min_center_distances_matrix = np.tile(min_center_distances, (self.k, 1))

        # print('center dist', center_distances, '\n\n', 'min dist', min_center_distances, '\n\n')
        
        membership_matrix = np.where(center_distances == min_center_distances_matrix, 1, 0)
        
        # print('selected memberhsip matrix', membership_matrix)
        
        if self.tie_breaker is True:
            membership = np.array([random.choice(np.where(membership_matrix[:,col] >0 )[0]) for col in range(self.n)])
        else:
            membership = membership_matrix.copy()

        return membership 
    
    def choose_centers(self, distance_matrix: np.ndarray, membership: np.ndarray, 
                       centers: np.ndarray) -> np.ndarray:
        """ 
        Choose centers as points which have the minimum total distance to all other points in cluster

        Args: 
            distance_matrix (np.ndarray): Distance between all pairs of points
            membership (np.ndarray) Array containing binary assignment matrix with a 1 at index i,j
                if point i belongs to cluster j
            centers (np.ndarray): Indices of cluster centers
        
        Returns:
            centers (np.ndarray) Updated indices of cluster centers
        """
        #Randomly assign points with multi-membership to a single cluster
        if self.tie_breaker is False:
            membership = np.array([random.choice(np.where(membership[:,col] >0 )[0]) for col in range(self.n)])

        # print('random membership', membership, '\n\n')

        for cluster in range(self.k):
            # print('randomly chosen', single_membership)
            # print("is this empty", np.where(single_membership == 0))

            #Get all members of a cluster
            # if self.tie_breaker is True:
            members = np.where(membership == cluster)[0]
            # else:
            #     members = np.where(membership[cluster] == 1)[0]

            #Calculate distance between that point and all members in the cluster
            member_distances = np.sum(distance_matrix[members, :][:, members], axis=0)
            min_distance = np.min(member_distances)
            minimal_members = np.where(member_distances == min_distance)[0]
            
            if len(member_distances) > 0:
                # points were assigned to that cluster
                if centers[cluster] in minimal_members:
                    #if the current center is actually a valid center
                    center_k = centers[cluster]

                else:
                    center_k = members[np.argmin(member_distances)]
                    print('not a valid center previously')

            centers[cluster] = center_k

        return centers
    
    def calculate_intra_cluster_distance(self, distance_matrix: np.ndarray,
                                         membership: np.ndarray, centers: np.ndarray) -> float:
        """
        Calculate the sum of the average distance between points and their cluster center. 

        Args:
            distance_matrix (np.ndarray): Distance between all pairs of points
            membership (np.ndarray) Array containing binary assignment matrix with a 1 at index i,j
                if point i belongs to cluster j
            centers (np.ndarray): Indices of cluster centers

        Returns:
            intra_cluster_sum (float): Sum of average distances of points to their centers
        """
        # #Find the distance from centers to members of their cluster 
        # point_distance_from_center = np.where(membership == 1, distance_matrix[centers, :], np.nan)
        
        # #Find average distance of members from their cluster centers
        # average_member_distance_from_center = np.nanmean(point_distance_from_center, axis = 1)
        
        # #Sum together average distances over all clusters
        # intra_cluster_sum = np.sum(average_member_distance_from_center)


        intra_cluster_sum = np.sum(np.where(membership == 1, distance_matrix[centers, :], 0))
        return intra_cluster_sum
    
    def first_kmeans(self, distance_matrix: np.ndarray):
        """
        Run k-means on the first time step.

        Args:
            distance_matrix (np.ndarray): Distance between all pairs of points
        
        Returns:
            membership (np.ndarray) Array containing binary assignment matrix with a 1 at index i,j
                if point i belongs to cluster j
            current_centers (np.ndarray): Indices of cluster centers
        """
        #Choose points that are most connected to all other points thorughout time as initial centers
        # init_centers = np.argsort(np.sum(np.sum(distance_matrix, axis=2), axis=0))[
        #     : self.k
        # ]

        # print('init centers', init_centers)
        # Choose points that are most connected to all other points at t0
        # init_centers = np.argsort(np.sum(distance_matrix[0], axis = 1))[:self.k]
        point_distances = np.sum(distance_matrix[0], axis = 1)
        min_point_distance = np.sort(point_distances)[self.k]
        potential_centers = np.where(point_distances <= min_point_distance)[0]
        init_centers = random.sample(list(potential_centers), self.k)
        init_centers = np.array(init_centers)

        # init_centers = np.zeros(self.k)
        
        # point_distances = np.sum(distance_matrix[0], axis = 1)
        # unique_point_distances = np.sort(np.unique(point_distances))[:self.k]
        # for index, unique_distance in enumerate(unique_point_distances):
        #     potential_centers = np.where(point_distances == unique_distance)[0]
        #     init_centers[index] = random.sample(list(potential_centers), 1)[0]

        # init_centers = init_centers.astype(int)

        # print('initial chosen centers', init_centers)
        init_matrix = distance_matrix[0]
        curr_centers = init_centers.copy()

        for iter in range(self.iter):
            membership = self.assign_points(distance_matrix = init_matrix, centers = curr_centers)
            new_centers = self.choose_centers(distance_matrix = init_matrix, membership = membership, centers = curr_centers)

            # print('new centers', new_centers)
            # print(membership, 'membership', new_centers, 'new_centers', new_centers,'\n\n')

            if (new_centers == curr_centers).all():
                return membership, curr_centers
            
            curr_centers = new_centers.copy()

            if iter == self.iter -1:
                print('reached max iterations')

        return membership, curr_centers
    
    def find_center_connections(self, current_centers: np.ndarray, distance_matrix: np.ndarray, time: int): 
        """
        Find centers connected at least p time steps with max drift between timesteps
        """

        drift_time_slices = distance_matrix[(time - self.center_connectivity): time]
        target_sum = self.max_drift*min(time, self.center_connectivity)

        center_connections = [np.where(np.sum(drift_time_slices[:, center], axis = 0) <= target_sum)[0] for center in current_centers]

        # previous_distance = distance_matrix[time-1]
        # center_connections = [
        #     np.where(previous_distance[center, :] <= self.max_drift)[0]
        #     for center in current_centers
        # ]

        return center_connections

    def next_assignment_proxy(self, current_centers, distance_matrix: np.ndarray, time: int):
        """
        Get assingment at time
        """

        center_connections = self.find_center_connections(current_centers = current_centers, 
                                                     distance_matrix = distance_matrix,
                                                     time = time)
        
        # center_connections = [
        #     np.where(previous_distance[center, :] <= self.max_drift)[0]
        #     for center in current_centers
        # ]

        current_distance = distance_matrix[time]

        # Preference to keep cluster centers the same
        current_membership = self.assign_points(distance_matrix=current_distance, centers = current_centers)
        min_sum = self.calculate_intra_cluster_distance(distance_matrix=current_distance, membership = current_membership, centers = current_centers)
        final_members = current_membership

        k_indices = np.arange(self.k)
        random.shuffle(k_indices)

        shuffled_centers = np.array(center_connections, dtype= object)[k_indices]

        for shuffled_index, center_k_possibilities in enumerate(shuffled_centers):
            if len(center_k_possibilities) > 1:
                for possibility in center_k_possibilities:
                    changing_centers = current_centers.copy()
                    changing_centers[k_indices[shuffled_index]] = possibility

                    #Ensure that there are k unique cluster centers to make it a valid possibility
                    if len(set(changing_centers)) == self.k:
                        membership = self.assign_points(distance_matrix=current_distance, centers = changing_centers)
                        curr_sum = self.calculate_intra_cluster_distance(distance_matrix = current_distance, centers = changing_centers, membership = membership)
                        
                        if curr_sum < min_sum:
                            min_sum = curr_sum
                            current_centers[k_indices[shuffled_index]] = possibility
                            final_members = membership

        # print('distances', np.sum(np.where(final_members == 1, current_distance[current_centers, :], 0), axis = 1))
        # print('counts', np.sum(np.where(final_members == 1, 1, 0), axis = 1))
        # checked_centers = self.choose_centers(distance_matrix = current_distance, membership = final_members,
        #                                                  centers = current_centers)
        return final_members, current_centers

    def next_assignment(self, current_centers: np.ndarray,
                        distance_matrix: np.ndarray,
                        time: int):
        """
        Assign points at every timestep t based on assignments at timestep t-1

        Args: 
            current_centers (np.ndarray): Indices of cluster centers at current timestep 

        Returns:
            final_members (np.ndarray):  Array containing binary assignment matrix with a 1 at index i,j
                if point i belongs to cluster j. Assignment for time t
            final_centers (np.ndarray): Indices of cluster centers at current timestep 
        """
        # Find all vertices that are within max_drift distance of each current center

        center_connections = self.find_center_connections(current_centers = current_centers,
                                                     distance_matrix = distance_matrix,
                                                     time = time)
        

        current_distance = distance_matrix[time]
        # center_connections = [
        #     np.where(previous_distance[center, :] <= self.max_drift)[0]
        #     for center in current_centers
        # ]

        # Preference to keep cluster centers the same
        current_membership = self.assign_points(distance_matrix=current_distance, centers = current_centers)        
        min_sum = self.calculate_intra_cluster_distance(distance_matrix=current_distance,
                                                        membership = current_membership,
                                                        centers = current_centers)
        final_members = current_membership
        final_centers = current_centers

        for center_combination in itertools.product(*center_connections):
            # all chosen centers are unique
            if len(set(center_combination)) == self.k:
                # This will iterate through every possible subset of centers
                membership = self.assign_points(distance_matrix=current_distance, centers = center_combination)
                total_sum = self.calculate_intra_cluster_distance(distance_matrix = current_distance, membership = membership, 
                                                                  centers = center_combination) 
                                
                # Return centers with smallest average distances from their members
                if total_sum < min_sum:
                    final_centers = center_combination
                    final_members = membership
                    min_sum = total_sum
        return final_members, final_centers

    def run_stgkm(self, method: Literal['full', 'proxy'] = 'full'):
        """
        Run STGkM.
        """
        print('Running stgkm')

        penalized_distance = self.penalize_distance()

        current_members, current_centers = self.first_kmeans(distance_matrix= penalized_distance)

        previous_distance = penalized_distance[0]

        if self.tie_breaker is True:
            self.full_assignments[0] = current_members
        else:   
            self.full_assignments[0:self.k, :] = current_members

        self.full_centers[0] = current_centers


        for time in range(1, self.t):
            if time%10 == 0:
                print('Processing time', time)

            if method == 'full':
                new_members, new_centers = self.next_assignment(
                    current_centers=current_centers,
                    distance_matrix = penalized_distance, 
                    time = time,
                    )
            elif method == 'proxy':
                new_members, new_centers = self.next_assignment_proxy(
                    current_centers=current_centers,
                    distance_matrix=penalized_distance,
                    time=time,
                )

            self.full_centers[time] = new_centers
            if self.tie_breaker is True:
                self.full_assignments[time] = new_members
            else:
                self.full_assignments[time*(self.k):(time+1)*self.k,:] = new_members

            current_centers = list(new_centers).copy()

        self.ltc = find_final_label_sc(weights=self.full_assignments.T, k=self.k)

        print('Finished running stgkm.')
        return None

def visualize_graph(
    connectivity_matrix: np.ndarray,
    labels=[],
    centers=[],
    color_dict={0: "red", 1: "gray", 2: "green", 3: "blue", -1: "cyan"},
    figsize = (10,10)
):
    t, n, _ = connectivity_matrix.shape

    if len(np.unique(labels)) > len(color_dict):
        raise Exception("Color dictionary requires more than 4 labels")

    g0 = nx.Graph(connectivity_matrix[0])
    g0.remove_edges_from(nx.selfloop_edges(g0))
    pos = nx.spring_layout(g0)

    for time in range(t):
        plt.figure(figsize = figsize)
        # No labels
        if len(labels) == 0:
            nx.draw(nx.Graph(connectivity_matrix[time]), pos=pos, with_labels=True)
        # Static long term labels
        elif len(labels) == n:
            g = nx.Graph(connectivity_matrix[time])
            g.remove_edges_from(nx.selfloop_edges(g))
            nx.draw(
                g,
                pos=pos,
                node_color=[color_dict[label] for label in labels],
                with_labels=True,
            )
        # Changing labels at each time step
        elif len(labels) == t:
            if len(centers) != 0:
                center_size = np.ones(n) * 300
                center_size[centers[time].astype(int)] = 500
                g = nx.Graph(connectivity_matrix[time])
                g.remove_edges_from(nx.selfloop_edges(g))
                nx.draw(
                    g,
                    pos=pos,
                    node_color=[color_dict[label] for label in labels[time]],
                    node_size=center_size,
                    with_labels=True,
                )
            else:
                g = nx.Graph(connectivity_matrix[time])
                g.remove_edges_from(nx.selfloop_edges(g))
                nx.draw(
                    g,
                    pos=pos,
                    node_color=[color_dict[label] for label in labels[time]],
                    with_labels=True,
                )

        plt.show()
