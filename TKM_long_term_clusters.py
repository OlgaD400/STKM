import numpy as np

def find_k_clusters(k: int,
                    criteria_mat: np.ndarray,
    threshold_change: float = 0.10,
    verbose: bool = False,
    ):
    """
    Find k long term clusters.
    """
    threshold = 0.70

    long_term_clusters = find_long_term_clusters(similarity_threshold=threshold, criteria_mat = criteria_mat)

    iters = 0
    while len(long_term_clusters) != k:
        long_term_clusters = find_long_term_clusters(
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
