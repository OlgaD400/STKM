from typing import List, Dict, Tuple
from collections.abc import Callable
import numpy as np
from sklearn.metrics import pairwise_distances, adjusted_mutual_info_score


def similarity_measure(x: np.ndarray, y: np.ndarray) -> float:
    """
    Calculate similarity between two vectors. Takes into account order of elements in arrays.

    Args:
        x (np.array): Vector
        y (np.array): Vector
    Returns:
        float: similarity measure
    """
    return np.sum(x == y) / len(x)


def similarity_matrix(weights: np.ndarray, similarity_function: Callable) -> np.ndarray:
    """
    Return similarity matrix where entry i,j contains the similarity of point assignment histories i and j.

    Args:
        weights (np.array): Point assignment histories.
        similarity_function (function): Function that defines the similarity measure between two arrays.
    Returns:
        sim_mat (np.array): Similarity matrix.
    """
    if len(weights.shape) == 3:
        assignments = np.argmax(weights, axis=2).T
    elif len(weights.shape) == 2:
        assignments = weights
    sim_mat = pairwise_distances(assignments, metric=similarity_function)

    return sim_mat


def find_k_clusters(
    k: int,
    criteria_mat: np.ndarray,
    threshold_change: float = 0.10,
    verbose: bool = False,
) -> List[List[int]]:
    """
    Find k long term clusters.

    Assign points to the same long-term cluster if the similarity between their assignment histories is above some threshold.
    Find the threshold that results in as close to k long-term clusters as possible.

    Args:
        k (int): The number of long term clusters to look for.
        criteria_mat (np.ndarray): The similarity matrix where entry i,j contains the similarity of point assignment histories i and j.
        threshold_change (float): The amount by which to increment the similarity threshold if k clusters are not found.
    Returns:
        best_long_term_clusters (List[List[int]]): Lists of indices of the points in each of the clusters.
    """
    threshold = 0.70

    long_term_clusters = find_long_term_clusters(
        similarity_threshold=threshold, criteria_mat=criteria_mat
    )
    best_long_term_clusters = long_term_clusters

    iters = 0
    min_cluster_difference = abs(len(long_term_clusters) - k)

    while len(long_term_clusters) != k:
        long_term_clusters = find_long_term_clusters(
            similarity_threshold=threshold,
            criteria_mat=criteria_mat,
        )

        if len(long_term_clusters) > k:
            threshold -= threshold_change
        elif len(long_term_clusters) < k:
            threshold += threshold_change

        cluster_difference = abs(len(long_term_clusters) - k)
        if cluster_difference < min_cluster_difference:
            best_long_term_clusters = long_term_clusters

        iters += 1

        if threshold > 1:
            if verbose is True:
                print("Could not find k clusters")
            threshold = 1
            long_term_clusters = find_long_term_clusters(
                similarity_threshold=threshold, criteria_mat=criteria_mat
            )
            criteria_mat = criteria_mat

            return long_term_clusters

        elif threshold < 0:
            if verbose is True:
                print("Could not find k clusters")
            threshold = 0
            long_term_clusters = find_long_term_clusters(
                similarity_threshold=threshold, criteria_mat=criteria_mat
            )
            criteria_mat = criteria_mat

            return long_term_clusters

        elif iters > 1 / threshold_change:
            if verbose is True:
                print("Could not find k clusters")
            return best_long_term_clusters

    if verbose is True:
        print("Threshold for ", k, " clusters: ", threshold)

    return best_long_term_clusters


def find_long_term_clusters(
    similarity_threshold: float = 0.70,
    criteria_mat: np.ndarray = None,
) -> List[List[int]]:
    """
    Find long term clusters given a similarity threshold.

    Assign points to the same long-term cluster if the similarity between their assignment histories is above some threshold.

    Args:
        similarity_threshold (float): The similarity threshold beyond which points are assigned to the same long-term cluster.
        criteria_mat (np.ndarray): The similarity matrix where entry i,j contains the similarity of point assignment histories i and j.
    Return:
        long_term_clusters (List[List[int]]): Lists of indices of the points in each of the clusters.
    """

    point_of_interest = 0
    long_term_clusters = []
    all_checked_ind = []

    max_iters = criteria_mat.shape[0]
    ids = np.arange(max_iters)
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
    return long_term_clusters


def find_optimal_threshold(
    weights: np.ndarray, k: int, true_labels: np.ndarray
) -> Tuple[float, float]:
    """
    Caluclate the AMI and total AMI scores given the true and predicted cluster assignments.

    Args:
        weights (np.ndarray):
        k (int):
        true_labels (np.ndarray):
    Returns:
        ami (float):
        tot_ami (float):
    """
    pred_labels = find_final_labels(weights, k, all_labels=[], counts=[])

    ami = adjusted_mutual_info_score(true_labels, pred_labels)

    if len(weights.shape) == 3:
        assignments = np.argmax(weights, axis=2).T
        t, n, k = weights.shape
    elif len(weights.shape) == 2:
        assignments = weights
        n, t = weights.shape

    tot_ami = adjusted_mutual_info_score(
        np.tile(true_labels, t), (assignments.T).reshape(t * n)
    )

    return ami, tot_ami


def map_ltc(mapping: Dict, ltc: List[List[int]]) -> List[List[int]]:
    """
    Re-index long term clusters so that the points correspond to the point order in the original dataset.

    Args:
        mapping (Dict): Dictionary mapping row number in the similarity matrix to the point order in the orignal datset
        ltc (List[List[int]]): Long term clusters where points are indexed by their row number in the similarity matrix.

    Returns:
        mapped_ltc (List[List[int]]): The long-term clusters re-indexed in terms of the original point order.
    """
    mapped_ltc = []
    for cluster in ltc:
        mapped_cluster = []
        for point in cluster:
            mapped_cluster.append(mapping[point])

        mapped_ltc.append(mapped_cluster)

    return mapped_ltc


def ltc_to_list(n: int, mapped_ltc: List[List[int]]) -> np.ndarray:
    """
    Turn lists of points in clusters into an array where each entry i contains the cluster label of point i.

    Args:
        n (int): Number of points
        mapped_ltc (List[List[int]]): Long-term clusters after re-indexing so that points correspond to the point order in the original dataset.

    Returns:
        pred_labels (np.ndarray): Array where entry i contains the cluster label of point i.
    """
    pred_labels = np.zeros(n)

    for i, cluster in enumerate(mapped_ltc):
        pred_labels[cluster] = i

    return pred_labels


def find_final_labels(
    weights: np.ndarray,
    k: int,
    all_labels: List,
    counts: List[int],
    trials: int = 5,
    tot_trials: int = 0,
):
    """
    Find the final long-term cluster assignments that are output the majority of the time when points' cluster assignment histories are shuffled.

    Args:
        weights (np.ndarray):
        k (int): Number of long-term clusters.
        all_labels (List): List of all the long-term cluster assignments discovered thus far.
        counts (List [int]): Number of times each set of long-term clusters shows up.
        trials (int): Number of trials to run.
        tot_trials (int): Trials that have been run so far.
    """
    tot_trials += trials

    if len(weights.shape) == 3:
        t, n, k = weights.shape
        weights = np.argmax(weights, axis=2).T
    elif len(weights.shape) == 2:
        n, t = weights.shape

    for trial in range(trials):
        # for each trial, shuffle the assignment histories and get ltc
        indices = np.arange(n)
        np.random.shuffle(indices)
        mapping = {np.arange(n)[i]: indices[i] for i in range(n)}
        criteria_mat = similarity_matrix(weights[indices, :])
        ltc = find_k_clusters(k=k, criteria_mat=criteria_mat)

        mapped_ltc = map_ltc(mapping, ltc)
        mapped_labels = ltc_to_list(n, mapped_ltc)

        matched = False

        # If the mapped labels correspond exactly to a previous label, increase the count on the previous label.
        # Else, add the mapped labels as a new, unique mapping with a count of 1
        if len(all_labels) != 0:
            for index, labels in enumerate(all_labels):
                ami = adjusted_mutual_info_score(mapped_labels, labels)
                if ami == 1:
                    # If the labels match up exactly
                    matched = True
                    counts[index] += 1
                    break

        if matched is False:
            all_labels.append(mapped_labels)
            counts.append(1)

    # If the majority of trials result in the same label, return that label.
    if np.max(counts) >= tot_trials // 2 + 1:
        return all_labels[np.argmax(counts)]

    # Else continue to run 2 more trials at a time until you reach a maximum of 20 trials.
    # Return whichever long-term cluster is output most often.
    else:
        if tot_trials < 20:
            return find_final_labels(
                weights,
                k,
                all_labels=all_labels,
                counts=counts,
                trials=2,
                tot_trials=tot_trials,
            )
        else:
            print("exceeded trials")
            return all_labels[np.argmax(counts)]
