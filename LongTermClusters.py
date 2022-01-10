"""Find long term clusters in temporal data based on how often points are clustered together at time snapshots."""

import numpy as np
import pandas as pd
from scipy.spatial.distance import pdist, squareform
from typing import List, Tuple, Union, Literal, Optional
from sklearn.metrics.cluster import adjusted_rand_score


def sim_func(u: np.ndarray, v: np.ndarray) -> float:
    """
    Similarity function for two clustering history vectors.

    The clustering history vectors contain values of zero at timesteps where points do not exist.
    Returns how many elements the arrays have in common divided by the length of the number of timesteps the histories
    both exist.

    Args:
        u (np.ndarray): An array containing the cluster assignment history of a point.
        v (np.ndarray): An array containing the cluster assignment history of a point.

    Returns:
        (float): Similarity score.  Number of labels the two clustering histories have in common
    """
    assert len(u) == len(v), "Vectors must be the same length."

    overlap = np.intersect1d(np.nonzero(u), np.nonzero(v))

    if len(overlap) == 0:
        return 0

    else:
        return len(np.where(u[overlap] == v[overlap])[0]) / len(overlap)


def calculate_criteria_matrix(swarm_df: pd.DataFrame) -> np.ndarray:
    """
    Calculate criteria matrix, where entry [i,j] contains the similarity score between the clustering histories of point i and point j. j.

    An intermediate running_labels_mat is created that stores the clustering histories of each data point.  This matrix
    contains zeros at time steps where points do not exist.

    Args:
        swarm_df (pd.DataFrame): Dataframe containing columns 'entity_id', 'lat', 'lon', 'iteration', 'swarm_id', 'time_ts'.

    Returns:
        criteria_mat (np.ndarray): Matrix containing similarity scores between clustering histories of points.
    """
    ids = np.unique(swarm_df["entity_id"])
    times = np.unique(swarm_df["time_ts"])

    running_labels_mat = np.zeros((len(ids), len(times)))

    for idx, elt in enumerate(ids):
        entity_mat = swarm_df[swarm_df["entity_id"] == elt]
        running_labels_mat[idx, entity_mat["iteration"]] = (
            entity_mat["cluster_label"].to_numpy() + 1
        )

    criteria_mat = np.diag(np.ones(len(ids))) + squareform(
        pdist(running_labels_mat, sim_func)
    )

    return criteria_mat


class LongTermClusters:
    """
    Identify long term clusters based on how often points are clustered together at time snapshots.

    Args:
        swarm_df (pd.DataFrame): Dataframe containing columns 'entity_id', 'lat', 'lon', 'iteration', 'swarm_id', 'time_ts'.

    Attributes:
        start_time (str): First time step in the data.
        end_time (str): Last time step in the data.
    """

    def __init__(self, swarm_df: pd.DataFrame):
        self.swarm_df = swarm_df
        self.start_time = np.unique(swarm_df["time_ts"])[0]
        self.end_time = np.unique(swarm_df["time_ts"])[-1]

    def find_long_term_clusters(
        self,
        time_interval: List[str],
        similarity_threshold: float = 0.70,
        criteria_mat: np.ndarray = None,
    ) -> Tuple[List[List], np.ndarray]:
        r"""
        Find long term clusters.

        We compare points’ cluster assignment histories
        .. math::
            c_i=[c_{i,1},c_{i,2},... ,c_{i,n}]
        on the time interval [t0, tn] to one another.  Points are classified as belonging to the same cluster if
        .. math::
            \frac{|c_i \cup c_j |}{|c_i|} \geq \sigma$ where $\sigma \in [0,1]

        Once cluster assignment histories are compared pairwise, the results are agglomerated. Each point is assigned
        to a moving cluster of points with which it travels. New points and their moving cluster members are merged
        to existing long-term clusters, based on which long-term cluster they share the greatest number of members with.

        Args:
            time_interval (List[str]): Time interval considered in calculating long term clusters
            similarity_threshold (float): Threshold for determining point similarity
            criteria_mat (np.ndarray): Matrix containing similarity scores between clustering histories of points.

        Returns:
            long_term_clusters (List[np.ndarray]): Final identified long term clusters.
            criteria_mat (np.ndarray): Matrix containing similarity scores between clustering histories of points.

        """
        time_mask = (self.swarm_df["time_ts"] >= time_interval[0]) & (
            self.swarm_df["time_ts"] <= time_interval[1]
        )

        swarm_df_subset = self.swarm_df[time_mask]

        ids = np.unique(swarm_df_subset["entity_id"])

        if criteria_mat is None:
            criteria_mat = calculate_criteria_matrix(swarm_df_subset)

        point_of_interest = 0
        long_term_clusters = []
        all_checked_ind = []

        max_iters = criteria_mat.shape[0]
        n = 0

        while (point_of_interest < criteria_mat.shape[0]) and (n < max_iters):
            ind = list(
                np.where(criteria_mat[point_of_interest, :] >= similarity_threshold)[0]
            )

            all_checked_ind += ind

            if point_of_interest == 0:
                long_term_clusters.append(ind)

            overlap = 0
            max_overlap_index = None

            for index, long_term_cluster in enumerate(long_term_clusters):
                new_overlap = len(np.intersect1d(ind, long_term_clusters[index]))

                if new_overlap > overlap:
                    overlap = new_overlap
                    max_overlap_index = index

            if max_overlap_index is None:
                long_term_clusters.append(ind)
            else:
                difference = np.setdiff1d(ind, long_term_clusters[max_overlap_index])

                long_term_clusters[max_overlap_index] += list(difference)

            fin_diff = np.setdiff1d(ids, np.unique(all_checked_ind))
            if len(fin_diff) > 0:
                point_of_interest = fin_diff[0]
            else:
                point_of_interest = criteria_mat.shape[0]
            n += 1
        return long_term_clusters, criteria_mat

    def find_k_clusters(
        self,
        k: int,
        time_interval: List[str],
        threshold_change: float = 0.10,
        verbose: bool = False,
    ) -> Union[Tuple[List[List[int]], float], None]:
        """
        Find k long term clusters.

        We identify long-term clusters based on how similar points' cluster assignment histories are. To find exactly k
        long-term clusters, we iterate through different values of the similarity threshold, lowering the value if too
        many and raising the value if too few clusters are identified, until k clusters are found. If k clusters cannot
        be found, the user is informed that no value of the similarity threshold gives the desired result.

        Args:
            k (int): number of clusters to find
            time_interval (List[str]): time interval considered for finding k clusters
            threshold_change (float): Incremental change in threshold.
            verbose (bool): Option to print the similarity threshold required for k clusters.

        Returns:
            long_term_clusters (List[List[int]]): Final long-term clusters.

        """
        threshold = 0.70

        long_term_clusters, criteria_mat = self.find_long_term_clusters(
            time_interval=time_interval, similarity_threshold=threshold
        )

        iters = 0
        while len(long_term_clusters) != k:
            long_term_clusters, criteria_mat = self.find_long_term_clusters(
                time_interval=time_interval,
                similarity_threshold=threshold,
                criteria_mat=criteria_mat,
            )

            if len(long_term_clusters) > k:
                threshold -= threshold_change
            elif len(long_term_clusters) < k:
                threshold += threshold_change

            iters += 1

            if iters > 1 / threshold_change or threshold > 1 or threshold < 0:
                if verbose is True:
                    print("Threshold change too large.")
                return None

        if verbose is True:
            print("Threshold for ", k, " clusters: ", threshold)

        return long_term_clusters

    def return_labels(self, long_term_clusters: List[int]) -> List[int]:
        """
        Return a list of the long-term cluster labels from the long-term clusters identified.

        Args:
            long_term_clusters (List[int]): Points belonging to each cluster.

        Returns:
            labels (List[int]): Long term cluster labels.

        """
        labels = np.zeros(len(np.unique(self.swarm_df["entity_id"])))
        for i, long_term_cluster in enumerate(long_term_clusters):
            labels[long_term_cluster] = i
        return list(labels)

    def average_trajectories(self, long_term_clusters: List[int]) -> np.ndarray:
        """
        Return array containing average trajectories of all detected long-term clusters.

        Args:
            long_term_clusters (List[int]): Points belonging to each cluster.:

        Returns:
            avg_trajectory (np.ndarray)

        """
        times = np.unique(self.swarm_df["time_ts"])
        avg_trajectory = np.zeros((len(long_term_clusters), len(times), 2))

        for index, long_term_cluster in enumerate(long_term_clusters):
            temp = self.swarm_df[self.swarm_df["entity_id"].isin(long_term_cluster)]

            for idx, time in enumerate(times):
                avg_trajectory[index, idx, :] = np.average(
                    temp[temp["time_ts"] == time][["lat", "lon"]].to_numpy(), axis=0
                )

        return avg_trajectory

    def calculate_r_score(
        self,
        true_labels: List[int],
        k: Optional[int] = None,
        threshold: float = 0.70,
        option: Literal["static_threshold", "k_clusters"] = "static_threshold",
    ) -> List[float]:
        """
        Calculate r score.

        We compare the true long-term clusters to the outputed long-term clusters for every subset of time
        intervals [0,𝑖] from the dataset.  This method can be carried out either by using a single static similarity
        threshold value or by seeking exaclty k clusters for every time interval [0,𝑖] . In cases where k clusters
        cannot be found, the r score is not reported.

        Args:
            true_labels (List[int]): True long-term cluster labels.
            threshold (float): Similarity threshold if option is static threshold.
            k (Optional[int]): number of clusters if 'k clusters' are being found
            option: Option for how clusters will be found.  Options are 'static_threshold' or 'k_clusters'.

        Returns:
            r_scores (List[int]): R scores for each time interval [0, time]

        """
        r_scores = []

        if option not in ("static_threshold", "k_clusters"):
            raise ValueError("Unknown Option")

        if option == "k_clusters" and k is None:
            raise ValueError("Need to specify number of clusters.")

        for time in np.unique(self.swarm_df["time_ts"]):

            if option == "static_threshold":
                long_term_clusters, criteria_mat = self.find_long_term_clusters(
                    time_interval=[self.start_time, time],
                    similarity_threshold=threshold,
                )
            else:
                long_term_clusters = self.find_k_clusters(
                    k=k, time_interval=[self.start_time, time], threshold_change=0.005
                )

            if long_term_clusters is None:
                r = None
            else:
                labels = np.zeros(len(np.unique(self.swarm_df["entity_id"])))

                for ii, long_term_cluster in enumerate(long_term_clusters):
                    labels[long_term_cluster] = ii

                r = adjusted_rand_score(true_labels, labels)

            r_scores.append(r)

        return r_scores
