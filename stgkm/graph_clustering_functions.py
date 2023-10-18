"""Implementation of STGkM"""
from typing import Literal, List, Optional
import itertools
import random
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from stkm.TKM_long_term_clusters import agglomerative_clustering


class STGKM:
    """Implement Spatiotemporal Graph k-means (STGkM)"""
    def __init__(self,
                 distance_matrix: np.ndarray,
                 penalty: float,
                 max_drift: int,
                 num_clusters: int,
                 iterations: int = 10,
                 drift_time_window: int = 1,
                 tie_breaker: bool = False):
        """
        Initialize STGkM.

        Args:
            distance_matrix (np.ndarray):  Distance between all pairs of vertices.
            penalty (float): Penalty to assign to disconnected vertices during pre-processing.
            max_drift (int): Maximum distance between cluster centers over time.
            num_clusters (int): Number of clusters for STGkM.
            iterations (int): Max. iterations for first k-means run.
            drift_time_window (int): Number of timesteps centers must remain within max_drift
                of one another.
            tie_breaker (bool): Whether to force unique vertex assignment.
        """
        self.penalty = penalty
        self.max_drift = max_drift
        self.k = num_clusters
        self.distance_matrix = distance_matrix
        self.center_connectivity = drift_time_window

        self.timesteps, self.num_vertices, _ = self.distance_matrix.shape

        self.full_centers = np.zeros((self.timesteps, self.k))
        self.tie_breaker = tie_breaker

        if self.tie_breaker is True:
            self.full_assignments = np.zeros((self.timesteps, self.num_vertices))
        else:
            self.full_assignments = np.zeros((self.timesteps*self.k, self.num_vertices))
        self.ltc = np.zeros(self.num_vertices)
        self.iter = iterations

    def penalize_distance(self):
        """
        Pre-processing step. Assign penalty distance to disconnected vertices.
        """
        penalized_distance = np.where(
            self.distance_matrix == np.inf, self.penalty, self.distance_matrix
        )
        return penalized_distance
    
    def assign_points(self, distance_matrix: np.ndarray, centers: np.ndarray) -> np.ndarray:
        """
        Assign each point to its closest cluster center.

        Args:
            distance_matrix (np.ndarray): Distance between all pairs of vertices
            centers (np.ndarray): Indices of cluster centers
        
        Returns:
            membership (np.ndarray) Array containing binary assignment matrix with a 1 at index i,j 
            if point i belongs to cluster j
        """
        center_distances = distance_matrix[centers, :]
        min_center_distances = np.min(center_distances, axis = 0)
        min_center_distances_matrix = np.tile(min_center_distances, (self.k, 1))
        membership_matrix = np.where(center_distances == min_center_distances_matrix, 1, 0)

        #If points are forced to belong to one, rather than multiple clusters
        if self.tie_breaker is True:
            membership = np.array([random.choice(np.where(
                membership_matrix[:,col] >0 )[0])
                for col in range(self.num_vertices)])
        else:
            membership = membership_matrix.copy()

        return membership
    
    def choose_centers(self, distance_matrix: np.ndarray, membership: np.ndarray,
                       centers: np.ndarray) -> np.ndarray:
        """ 
        Choose centers as points which have the minimum total distance to all 
        other points in cluster

        Args: 
            distance_matrix (np.ndarray): Distance between all pairs of vertices
            membership (np.ndarray) Array containing binary assignment matrix with a 1 at index i,j
                if point i belongs to cluster j
            centers (np.ndarray): Indices of cluster centers
        
        Returns:
            centers (np.ndarray) Updated indices of cluster centers
        """
        if self.tie_breaker is False:
            #Randomly assign points with multi-membership to a single cluster
            membership = np.array([random.choice(
                np.where(
                    membership[:,col] >0 )[0])
                    for col in range(self.num_vertices)])

        for cluster in range(self.k):
            members = np.where(membership == cluster)[0]

            #Calculate distance between each point and all other members in the cluster
            member_distances = np.sum(distance_matrix[members, :][:, members], axis=0)
            min_distance = np.min(member_distances)
            #All members that are min distance away from the other points in the cluster
            minimal_members = np.where(member_distances == min_distance)[0]
            
            if len(member_distances) > 0:
                # points were assigned to that cluster
                if centers[cluster] in minimal_members:
                    #if the current center is the min distance away from
                    # other points in the cluster
                    center_k = centers[cluster]

                else:
                    #Reassign the current cluster center to be one of the points that are min
                    #distance away from the others in the cluster
                    center_k = members[np.argmin(member_distances)]
                    print('Not a valid center previously.')

            centers[cluster] = center_k

        return centers
    
    def calculate_intra_cluster_distance(self, distance_matrix: np.ndarray,
                                         membership: np.ndarray, centers: np.ndarray) -> float:
        """
        Calculate the sum of the average distance between vertices and their cluster center. 

        Args:
            distance_matrix (np.ndarray): Distance between all pairs of vertices
            membership (np.ndarray) Array containing binary assignment matrix with a 1 at index i,j
                if point i belongs to cluster j
            centers (np.ndarray): Indices of cluster centers

        Returns:
            intra_cluster_sum (float): Sum of average distances of points to their centers
        """

        intra_cluster_sum = np.sum(np.where(membership == 1, distance_matrix[centers, :], 0))
        return intra_cluster_sum
    
    def first_kmeans(self, distance_matrix: np.ndarray):
        """
        Run k-means on the first time step.

        Args:
            distance_matrix (np.ndarray): Distance between all pairs of vertices
        
        Returns:
            membership (np.ndarray) Array containing binary assignment matrix with a 1 at index i,j
                if point i belongs to cluster j
            current_centers (np.ndarray): Indices of cluster centers
        """

        #Sample initial centers from vertices which are closest to all other points
        #in the first time step
        point_distances = np.sum(distance_matrix[0], axis = 1)
        min_point_distance = np.sort(point_distances)[self.k]
        potential_centers = np.where(point_distances <= min_point_distance)[0]
        init_centers = random.sample(list(potential_centers), self.k)
        init_centers = np.array(init_centers)

        init_matrix = distance_matrix[0]
        curr_centers = init_centers.copy()

        for iterations in range(self.iter):
            membership = self.assign_points(
                distance_matrix = init_matrix,
                centers = curr_centers)

            new_centers = self.choose_centers(
                distance_matrix = init_matrix,
                membership = membership,
                centers = curr_centers)

            #If there is no difference between the current and the previously
            #predicted centers
            if (new_centers == curr_centers).all():
                #Why am I doing this here? I had a bug where membership was
                #wrong when I got here, but that doesn't make sense.
                # This fixed it though.
                membership = self.assign_points(
                    distance_matrix = init_matrix,
                    centers = new_centers)

                return membership, curr_centers
            
            curr_centers = new_centers.copy()

            if iterations == self.iter -1:
                print('reached max iterations')

        return membership, curr_centers
    
    def find_center_connections(self, 
                                current_centers: np.ndarray,
                                distance_matrix: np.ndarray,
                                time: int) -> List[List[int]]:
        """
        Find centers connected at least "center connectivity" time steps with no 
        more than "max drift" distance between timesteps.

        Args:
            current_centers (np.ndarray): Current cluster centers.
            distance_matrix (np.ndarray): Distances between all pairs of vertices.
            time (int): Current time step.

        Returns:
            center_connections List[List[int]]: Each entry contains list of all vertices 
                connected to previous center.
        """
        #Distances matrices for "center connectivity" previous time steps
        drift_time_slices = distance_matrix[(time - self.center_connectivity): time]
        #The sum across the "center connectivity" previous distance matrices can be no
        #larger than target sum
        target_sum = self.max_drift*min(time, self.center_connectivity)
    
        center_connections = [np.where(
            np.sum(drift_time_slices[:, center], axis = 0)
            <= target_sum)[0]
            for center in current_centers]

        return center_connections

    def next_assignment_proxy(self, current_centers, distance_matrix: np.ndarray, time: int):
        """
        Assign points to new cluster centers at the current time using the approximate approach.

        Args:
            current_centers (np.ndarray): Current cluster centers.
            distance_matrix (np.ndarray): Distances between all pairs of vertices.
            time (int): Current time step. 

        Returns:
            final_members (np.ndarray): Cluster membership for new centers. 
            current_centers (np.ndarray): New centers at current time.
        """
        #Find all potential new cluster centers based on "cluster connectivity" and "max drift"
        center_connections = self.find_center_connections(
            current_centers = current_centers,
            distance_matrix = distance_matrix,
            time = time)

        current_distance = distance_matrix[time]

        # Preference to keep cluster centers the same
        current_membership = self.assign_points(
            distance_matrix=current_distance,
            centers = current_centers)
        min_sum = self.calculate_intra_cluster_distance(
            distance_matrix=current_distance,
            membership = current_membership,
            centers = current_centers)
        final_members = current_membership

        #Shuffle the potential new centers, so they're processed in a random order
        k_indices = np.arange(self.k)
        random.shuffle(k_indices)
        shuffled_centers = np.array(center_connections, dtype= object)[k_indices]

        for shuffled_index, center_k_possibilities in enumerate(shuffled_centers):
            #If there's more than a single possibility for this cluster center
            if len(center_k_possibilities) > 1:
                #Keep all other cluster centers fixed, while updating one
                for possibility in center_k_possibilities:
                    changing_centers = current_centers.copy()
                    changing_centers[k_indices[shuffled_index]] = possibility

                    #Ensure that there are k unique cluster centers to make it a valid possibility
                    if len(set(changing_centers)) == self.k:
                        membership = self.assign_points(
                            distance_matrix=current_distance,
                            centers = changing_centers)
                        curr_sum = self.calculate_intra_cluster_distance(
                            distance_matrix = current_distance,
                            centers = changing_centers,
                            membership = membership)
                        
                        if curr_sum < min_sum:
                            min_sum = curr_sum
                            current_centers[k_indices[shuffled_index]] = possibility
                            final_members = membership

        return final_members, current_centers

    def next_assignment(self, current_centers: np.ndarray,
                        distance_matrix: np.ndarray,
                        time: int):
        """
        Assign points at current time.

        Args: 
            current_centers (np.ndarray): Indices of cluster centers at current timestep 
            distance_matrix (np.ndarray): Distance between all pairs of vertices.
            time (int): Current time. 

        Returns:
            final_members (np.ndarray):  Binary assignment matrix with a 1 at index i,j
                if point i belongs to cluster j. Assignment for current time. 
            final_centers (np.ndarray): Indices of cluster centers at current timestep 
        """
        #Find all potential new cluster centers based on "cluster connectivity" and "max drift"
        center_connections = self.find_center_connections(
            current_centers = current_centers,
            distance_matrix = distance_matrix,
            time = time)
        #Preference to keep cluster centers the same
        current_distance = distance_matrix[time]
        current_membership = self.assign_points(
            distance_matrix=current_distance,
            centers = current_centers)
        min_sum = self.calculate_intra_cluster_distance(
            distance_matrix=current_distance,
            membership = current_membership,
            centers = current_centers)
        final_members = current_membership
        final_centers = current_centers

        #Find all possible combinations of cluster centers
        for center_combination in itertools.product(*center_connections):
            # All chosen centers are unique
            if len(set(center_combination)) == self.k:
                membership = self.assign_points(
                    distance_matrix=current_distance,
                    centers = center_combination)
                total_sum = self.calculate_intra_cluster_distance(
                    distance_matrix = current_distance,
                    membership = membership,
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

        Args:
            method (Literal['full', 'proxy']): Whether to run STGkM with optimal or approximate assignment.
        """
        print('Running stgkm')
        
        #Pre-process the data
        penalized_distance = self.penalize_distance()

        #First k-means
        current_members, current_centers = self.first_kmeans(distance_matrix= penalized_distance)

        #If single-membership is enforced vs not
        if self.tie_breaker is True:
            self.full_assignments[0] = current_members
        else:   
            self.full_assignments[0:self.k, :] = current_members

        self.full_centers[0] = current_centers

        for time in range(1, self.timesteps):
            if time%10 == 0:
                print('Processing time', time)

            #Update members and centers using either the full or approximate approach
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

            #If single membership is enforced vs not
            if self.tie_breaker is True:
                self.full_assignments[time] = new_members
            else:
                self.full_assignments[time*(self.k):(time+1)*self.k,:] = new_members

            current_centers = list(new_centers).copy()

        #Find long term clusters based on total assignment history
        self.ltc = agglomerative_clustering(weights=self.full_assignments.T, k=self.k)

        print('Finished running stgkm.')
        return None

def visualize_graph(
    connectivity_matrix: np.ndarray,
    labels: Optional[List] = None,
    centers: Optional[List] = None,
    color_dict: Optional[dict] = None,
    figsize = (10,10)
):
    """
    Visualize the dynamic graph at each time step. 
    """
    timesteps, num_vertices, _ = connectivity_matrix.shape

    if labels is None:
        labels = []
    if centers is None:
        centers = []
    if color_dict is None:
        color_dict = {0: "red", 1: "gray", 2: "green", 3: "blue", -1: "cyan"}

    if len(np.unique(labels)) > len(color_dict):
        raise Exception("Color dictionary requires more than 4 keys/values")

    #Set layout for figures
    g_0 = nx.Graph(connectivity_matrix[0])
    g_0.remove_edges_from(nx.selfloop_edges(g_0))
    pos = nx.spring_layout(g_0)

    for time in range(timesteps):
        plt.figure(figsize = figsize)
        # No labels
        if len(labels) == 0:
            nx.draw(nx.Graph(connectivity_matrix[time]), pos=pos, with_labels=True)
        # Static long term labels
        elif len(labels) == num_vertices:
            graph = nx.Graph(connectivity_matrix[time])
            graph.remove_edges_from(nx.selfloop_edges(graph))
            nx.draw(
                graph,
                pos=pos,
                node_color=[color_dict[label] for label in labels],
                with_labels=True,
            )
        # Changing labels at each time step
        elif len(labels) == timesteps:
            if len(centers) != 0:
                center_size = np.ones(num_vertices) * 300
                center_size[centers[time].astype(int)] = 500
                graph = nx.Graph(connectivity_matrix[time])
                graph.remove_edges_from(nx.selfloop_edges(graph))
                nx.draw(
                    graph,
                    pos=pos,
                    node_color=[color_dict[label] for label in labels[time]],
                    node_size=center_size,
                    with_labels=True,
                )
            else:
                graph = nx.Graph(connectivity_matrix[time])
                graph.remove_edges_from(nx.selfloop_edges(graph))
                nx.draw(
                    graph,
                    pos=pos,
                    node_color=[color_dict[label] for label in labels[time]],
                    with_labels=True,
                )

        plt.show()
