import numpy as np
from sklearn.metrics import pairwise_distances, adjusted_mutual_info_score, f1_score


def fill_out_predictions(assignments, t):
    n, t_size = assignments.shape
    new_assignments = np.zeros((n, t))

    for j, row in enumerate(assignments):
        for i, entry in enumerate(row[:-1]):
            if row[i] == row[i + 1]:
                new_assignments[j, i * 10:(i + 1) * 10] = entry
                print(new_assignments[j, :])
            else:
                new_assignments[j, i * 10:(i + 1) * 10 // 2] = entry
                new_assignments[j, (i + 1) * 10 // 2:(i + 1) * 10] = row[i + 1]
    return new_assignments


def similarity_measure(x: np.array, y: np.array) -> np.float:
    """
    Calculate similarity between two vectors.

    Args:
        x (np.array): Vector
        y (np.array): Vector
    Returns:
        np.float: similarity measure
    """
    return np.sum(x == y)/len(x)


def similarity_matrix(weights: np.array):
    """
    Return similarity matrix where entry i,j contains the similarity of point assignment histories i and j.

    Args:
        weights (np.array): Point assignment histories.
    Returns:
        sim_mat (np.array): Similarity matrix.
    """
    if len(weights.shape) == 3:
        assignments = np.argmax(weights, axis=2).T
    elif len(weights.shape) == 2:
        assignments = weights
    sim_mat = pairwise_distances(assignments, metric=similarity_measure)

    return sim_mat


def find_k_clusters(k: int,
                    criteria_mat: np.ndarray,
    threshold_change: float = 0.10,
    verbose: bool = False,
    ):
    """
    Find k long term clusters.
    """
    threshold = 0.70

    long_term_clusters = find_long_term_clusters(similarity_threshold=threshold, criteria_mat=criteria_mat)
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
            print('Could not find k clusters')
            threshold = 1
            long_term_clusters = find_long_term_clusters(similarity_threshold=threshold, criteria_mat=criteria_mat)
            criteria_mat = criteria_mat

            return long_term_clusters

        elif threshold < 0:
            print('Could not find k clusters')
            threshold = 0
            long_term_clusters = find_long_term_clusters(similarity_threshold=threshold, criteria_mat=criteria_mat)
            criteria_mat = criteria_mat

            return long_term_clusters

    if verbose is True:
        print("Threshold for ", k, " clusters: ", threshold)

    return best_long_term_clusters



def find_long_term_clusters(
    similarity_threshold: float = 0.70,
    criteria_mat: np.ndarray = None,
):
    """
    Find long term clusters.
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


def find_optimal_threshold(weights, k, true_labels):
    n,t = weights.shape
    criteria_mat = similarity_matrix(weights)

    ltc = find_k_clusters(k=k, criteria_mat=criteria_mat, threshold_change=.05)

    # if ltc is None:
    #     thresholds = np.linspace(.10, .90, 9)
    #     ami = 0
    #
    #     for threshold in thresholds:
    #         ltc = find_long_term_clusters(similarity_threshold=threshold, criteria_mat=criteria_mat)
    #
    #         pred_labels = np.zeros(n)
    #
    #         for i, cluster in enumerate(ltc):
    #             pred_labels[cluster] = i
    #
    #         current_ami = adjusted_mutual_info_score(true_labels, pred_labels)
    #
    #         if current_ami > ami:
    #             ami = current_ami
    #
    # else:
    pred_labels = np.zeros(n)

    for i, cluster in enumerate(ltc):
        pred_labels[cluster] = i

    ami = adjusted_mutual_info_score(true_labels, pred_labels)

    assignments = weights

    tot_ami = adjusted_mutual_info_score(np.tile(true_labels, t), (assignments.T).reshape(t * n))

    # f1 = f1_score(np.tile(true_labels, t), (assignments.T).reshape(t * n))

    return ami, tot_ami