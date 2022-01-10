"""Method for finding moving clusters in consecutive time snapshots."""

import pandas as pd
import numpy as np
from scipy.sparse import coo_matrix, block_diag, bmat
from sklearn.cluster import DBSCAN
import networkx as nx
from sklearn.metrics.cluster import adjusted_rand_score
from typing import List, Tuple, Union


def run_dbscan(swarm_df: pd.DataFrame, eps: float) -> List[np.ndarray]:
    """
    Run DBSCAN at every time step and output labels.  Update the swarm dataframe to contain the cluster labels of each point at every time step in a column "cluster_label".

    Args:
        swarm_df (pd.DataFrame): Dataframe containing columns 'entity_id', 'lat', 'lon', 'iteration', 'swarm_id', 'time_ts'.
        eps (float): Value of epsilon for DBSCAN, which controls the radius of the neighborhood for points to be classified as belonging to the same cluster.

    Returns:
        labels (List[np.ndarray]): List of labels of DBSCAN at every time step.
    """
    labels = []

    times = np.unique(swarm_df["time_ts"])

    for time in times:
        time_array = swarm_df[swarm_df["time_ts"] == time][["lat", "lon"]].to_numpy()
        clustering = DBSCAN(eps=eps, min_samples=3).fit(time_array)
        classes, class_ind = np.unique(clustering.labels_, return_inverse=True)
        labels.append(class_ind)

    flat_labels = [item for sublist in labels for item in sublist]
    swarm_df["cluster_label"] = flat_labels

    return labels


class ConsecutiveClusters:
    """
    Create a new class for spatio-temporal clustering.

    Identify moving clusters of points between consecutive time steps using the moving cluster criteria proposed by Kalnis et. al.
    Given a sequence of clusters c_{i,t} at every time snapshot t, c_{i,t} c_{i,t+1} is a moving cluster if |c_{i,t}  ∩ c_{i,t+1}|/|c_(i,t)  ∪ c_{i,t+1}| ≥θ where θ ∈[0,1] is an integrity threshold for the contents of the two clusters.

    Args:
        swarm_df (pd.DataFrame): Dataframe containing columns 'entity_id', 'lat', 'lon', 'iteration', 'swarm_id',
        'time_ts', and 'cluster_label'.
        threshold (float): Similarity threshold for the contents of two clusters at consecutive time snapshots.

    Attributes:
        graph_df (pd.DataFrame): Dataframe containing columns 'source', 'target', 'weight', 'true_source', 'true_target', 'start_time', 'end_time'.
        graph (nx.graph.Graph) networkX graph: Graph.
    """

    def __init__(self, swarm_df: pd.DataFrame, threshold: float):
        self.swarm_df = swarm_df
        self.threshold = threshold
        self.graph_df = None
        self.graph = None

        assert (threshold <= 1) and (
            threshold >= 0
        ), "Threshold value must be between 0 and 1."

    def create_graph(self) -> None:
        """
        Create NetworkX graph from moving clusters, create dataframe storing graph nodes and edge weights.

        Graph dataframe contains columns 'source', 'target', 'weight', 'true_source', 'true_target', 'start_time', 'end_time'.
        """
        times = np.unique(self.swarm_df["time_ts"])

        assert (
            len(times) > 1
        ), "Need data from more than one time step to create a graph."

        off_diag_mats = []
        true_clusters = []
        timestamps = []

        curr_df = self.swarm_df[self.swarm_df["time_ts"] == times[0]]
        classes_padding = len(np.unique(curr_df["cluster_label"]))

        true_clusters.append(np.arange(classes_padding))
        timestamps.append([times[0]] * classes_padding)
        num_curr_clusters = 0

        for i in range(1, len(times)):
            prev_df = self.swarm_df[self.swarm_df["time_ts"] == times[i - 1]]
            curr_df = self.swarm_df[self.swarm_df["time_ts"] == times[i]]

            num_curr_clusters = len(np.unique(curr_df["cluster_label"]))
            true_clusters.append(np.arange(num_curr_clusters))
            timestamps.append([times[i]] * num_curr_clusters)

            criteria_mat = np.zeros(
                (len(np.unique(prev_df["cluster_label"])), num_curr_clusters)
            )

            for index_1, label_prev in enumerate(np.unique(prev_df["cluster_label"])):
                for index_2, label_curr in enumerate(
                    np.unique(curr_df["cluster_label"])
                ):
                    prev_members = prev_df[prev_df["cluster_label"] == label_prev][
                        "entity_id"
                    ].tolist()
                    curr_members = curr_df[curr_df["cluster_label"] == label_curr][
                        "entity_id"
                    ].tolist()

                    criteria = len(np.intersect1d(prev_members, curr_members)) / len(
                        np.unique(prev_members + curr_members)
                    )
                    criteria_mat[index_1, index_2] = criteria

            criteria_mat[criteria_mat < self.threshold] = 0

            criteria_mat = coo_matrix(criteria_mat)

            off_diag_mats.append(criteria_mat)

        off_diagonal = block_diag(off_diag_mats)

        left_padding = coo_matrix((off_diagonal.shape[0], classes_padding))
        lower_padding = coo_matrix((num_curr_clusters, classes_padding))

        upper_tri = bmat([[left_padding, off_diagonal], [lower_padding, None]])
        upper_tri.eliminate_zeros()

        graph_data = {
            "source": upper_tri.row,
            "target": upper_tri.col,
            "weight": upper_tri.data,
        }

        self.graph_df = pd.DataFrame(data=graph_data)

        self.graph = nx.from_pandas_edgelist(
            self.graph_df,
            source="source",
            target="target",
            edge_attr="weight",
            create_using=nx.DiGraph(),
        )

        flat_true_clusters = [item for sublist in true_clusters for item in sublist]
        self.graph_df["true_source"] = np.array(flat_true_clusters)[
            self.graph_df["source"]
        ]
        self.graph_df["true_target"] = np.array(flat_true_clusters)[
            self.graph_df["target"]
        ]

        flat_timestamps = [item for sublist in timestamps for item in sublist]
        self.graph_df["start_time"] = np.array(flat_timestamps)[self.graph_df["source"]]
        self.graph_df["end_time"] = np.array(flat_timestamps)[self.graph_df["target"]]

    def find_successors(self, start_node: int) -> List[int]:
        """
        Find the immediate successors of a node in the graph.

        Args:
            start_node (int): Node to search for successors from.

        Returns:
            successors (List[int]) : List of immediate successors of start_node.

        """
        source_df = self.graph_df[self.graph_df["source"] == start_node]
        successors = source_df["target"].tolist()

        return successors

    def find_labels_at_t(self, t: str) -> Union[np.ndarray, None]:
        """
        Identify moving cluster labels at time t in the graph.

        Args:
            t (str): Time from self.graph_df column "start_time" or "end_time".

        Returns:
            labels (np.ndarray): list of moving cluster labels.
        """
        num_points = len(np.unique(self.swarm_df["entity_id"]))
        labels = np.zeros(num_points)

        if t not in np.unique(self.graph_df["start_time"]) and t not in np.unique(
            self.graph_df["end_time"]
        ):
            print("No moving clusters at time ", t)
            return None

        graph_time = self.graph_df[self.graph_df["start_time"] == t]

        if len(graph_time) == 0:
            graph_time = self.graph_df[self.graph_df["end_time"] == t]

            for idx, row in graph_time.iterrows():
                indices = (
                    self.swarm_df[
                        (self.swarm_df["time_ts"] == t)
                        & (self.swarm_df["cluster_label"] == row["true_target"])
                    ]["entity_id"]
                    .to_numpy()
                    .astype(int)
                )
                labels[indices] = idx + 1

        else:
            for idx, row in graph_time.iterrows():
                indices = (
                    self.swarm_df[
                        (self.swarm_df["time_ts"] == t)
                        & (self.swarm_df["cluster_label"] == row["true_source"])
                    ]["entity_id"]
                    .to_numpy()
                    .astype(int)
                )
                labels[indices] = idx + 1

        return labels

    def find_node_rows(
        self, node: int
    ) -> Union[Tuple[List[int], bool], Tuple[None, bool]]:
        """
        Find rows in swarm_df associated with the cluster at a graph node.

        Args:
            node (int): Graph node.

        Returns:
            node_rows(list[int]): list of dataframe rows corresponding to points that are part of the cluster at the given node.
            end_of_moving_cluster (bool): Boolean value determining whether or not the node is the last in a moving cluster path.
        """
        node_rows = None
        end_of_moving_cluster = False
        source_df = self.graph_df[self.graph_df["source"] == node]

        if len(source_df) != 0:
            node_rows = self.swarm_df[
                (self.swarm_df["cluster_label"] == source_df.iloc[0]["true_source"])
                & (self.swarm_df["time_ts"] == source_df.iloc[0]["start_time"])
            ].index
        else:
            target_df = self.graph_df[self.graph_df["target"] == node]

            if len(target_df) != 0:
                node_rows = self.swarm_df[
                    (self.swarm_df["cluster_label"] == target_df.iloc[0]["true_target"])
                    & (self.swarm_df["time_ts"] == target_df.iloc[0]["end_time"])
                ].index
            end_of_moving_cluster = True

        return node_rows, end_of_moving_cluster

    def track_cluster_path_df(
        self, start_nodes: List[int], location_history: List[np.ndarray]
    ) -> Union[pd.DataFrame, None]:
        """
        Track the path of a moving cluster from the graph. Return a dataframe with a column specifying which rows are part of the moving cluster.

        Args:
            start_nodes (List[int]): Nodes to start tracking the cluster path from.
            location_history (List[np.ndarray]): List storing the cluster path history.  Each array contains the row numbers of the points that were part of the moving cluster at a given time step.

        Returns:
            cluster_path_df (pd.DataFrame): Dataframe containing column with labels determining whether points are part of moving cluster or not.

        """
        successors = []
        row_list = []

        for start_node in start_nodes:
            rows, end_of_moving_cluster = self.find_node_rows(start_node)
            if rows is None:
                print("Not a moving cluster.")
                return None
            else:
                row_list.append(rows)

            if end_of_moving_cluster is False:
                successors += self.find_successors(start_node)

        location_history.append(np.concatenate(row_list))

        if len(successors) > 0:
            location_history = self.track_cluster_path_df(
                list(np.unique(successors)), location_history=location_history
            )
        else:
            location_history = np.concatenate(location_history)
            moving_cluster_col = np.zeros(len(self.swarm_df))
            moving_cluster_col[location_history] = 1

            cluster_path_df = self.swarm_df.assign(
                moving_cluster_col=moving_cluster_col
            )

            return cluster_path_df

        return location_history

    def calculate_r_scores(self, true_labels: List[int]) -> List[int]:
        """
        Calculate R score for moving clusters at every time step.

        Args:
            true_labels (List[int]): True moving cluster labels.

        Returns:
            r_scores (List[int]): R scores over all of the time steps.
        """
        r_scores = []

        for t in np.unique(self.swarm_df["time_ts"]):
            labels = self.find_labels_at_t(t)

            if labels is None:
                r_scores.append(0)
            else:
                r = adjusted_rand_score(true_labels, labels)
                r_scores.append(r)

        return r_scores
