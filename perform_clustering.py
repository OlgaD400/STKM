from TKM import TKM
from TKM_long_term_clusters import find_long_term_clusters, find_k_clusters, similarity_matrix
from sklearn.metrics.cluster import adjusted_mutual_info_score
from sklearn.cluster import KMeans, DBSCAN
import time

import numpy as np
import pandas as pd


def read_data(csv_path: str, min_size=0, max_size=3000):
    """
    Read the data from the csv file.

    Return the data both in animation format in in clustering format.

    Args:
        csv_path (str): Path to csv file containing data.
        min_size (int): Minimum dataset size
        max_size (int): Maximum dataset size

    Returns:
        data (pd.DataFrame): The dataframe from the csv file.
        clustering_data (np.array): The data formatted for trimmed k-means clustering,
            shaped txmxn.
        true_labels (List): The ground truth long-term labels of each point.
    """

    data = pd.read_csv(csv_path)

    clustering_data = None
    true_labels = None

    dataset_size = data['id'].nunique()*data['frame'].nunique()

    if dataset_size in range(min_size, max_size):
        data['x'] = (data['x'] - data['x'].min()) / (data['x'].max() - data['x'].min())
        data['y'] = (data['y'] - data['y'].min()) / (data['y'].max() - data['y'].min())

        frame_positions = []

        for frame in data['frame'].unique():
            frame_data = data[data['frame'] == frame]

            n_id = frame_data['id'].nunique()

            frame_data["id_cumcounts"] = frame_data.groupby("id").cumcount()

            frame_data.sort_values(["id_cumcounts", "id"], inplace=True)

            frame_positions.append(frame_data[['x', 'y']].to_numpy()[:n_id])

        true_labels = frame_data['cid'].to_list()[:n_id]

        try:
            clustering_data = np.transpose(np.array(frame_positions), axes=[0, 2, 1])

        except:
            print(csv_path, ' unable to be processed')

    return data, clustering_data, true_labels


def perform_clustering(clustering_data: np.array, true_labels: np.array, lam: float = .60, max_iter: int = 5000) -> np.float:
    """
    Perform clustering and evaluate AMI against the ground-truth long-term clusters.

    Args:
        clustering_data (np.array): Data to be clustered.
        true_labels (np.array): Ground truth long-term clusters.
        lam (float): Tuning parameter
        max_iter (int): Maximum number of iterations for tkm.

    Returns:
        ami (float): Adjusted mutual info score.
    """

    t, m, n = clustering_data.shape
    k = len(np.unique(true_labels))

    tkm = TKM(clustering_data)
    
    start_time = time.time()
    tkm.perform_clustering(k=k, lam=lam, max_iter=max_iter)
    runtime = time.time() - start_time


    criteria_mat = similarity_matrix(tkm.weights)

    ltc = find_k_clusters(k=k, criteria_mat=criteria_mat, threshold_change=.05)

    if ltc is None:
        thresholds = np.linspace(.10, .90, 9)
        ami = 0

        for threshold in thresholds:
            ltc = find_long_term_clusters(similarity_threshold=threshold, criteria_mat=criteria_mat)

            pred_labels = np.zeros(n)

            for i, cluster in enumerate(ltc):
                pred_labels[cluster] = i

            current_ami = adjusted_mutual_info_score(true_labels, pred_labels)

            if current_ami > ami:
                ami = current_ami

    else:
        pred_labels = np.zeros(n)

        for i, cluster in enumerate(ltc):
            pred_labels[cluster] = i

        ami = adjusted_mutual_info_score(true_labels, pred_labels)

    assignments = np.argmax(tkm.weights, axis=2).T

    tot_ami = adjusted_mutual_info_score(np.tile(true_labels, t), (assignments.T).reshape(t*n))

    return ami, tot_ami, runtime, tkm.weights


def k_means(clustering_data: np.array, true_labels: np.array) -> np.float:
    """
    Perform standard k_means clustering and evaluate AMI against the ground-truth long-term clusters.

    Args:
        clustering_data (np.array): Data to be clustered.
        true_labels (np.array): Ground truth long-term clusters.
        max_iter (int): Maximum number of iterations for tkm.

    Returns:
        ami (float): Adjusted mutual info score.
    """
    t, m, n = clustering_data.shape
    k = len(np.unique(true_labels))

    weights = []
    for time in range(t):
        weights.append(KMeans(n_clusters=k).fit_predict(clustering_data[time, :,:].T))

    weights = np.array(weights).T

    criteria_mat = similarity_matrix(weights)

    ltc = find_k_clusters(k=k, criteria_mat=criteria_mat, threshold_change=.05)

    if ltc is None:
        thresholds = np.linspace(.10, .90, 9)
        ami = 0

        for threshold in thresholds:
            ltc = find_long_term_clusters(similarity_threshold=threshold, criteria_mat=criteria_mat)

            pred_labels = np.zeros(n)

            for i, cluster in enumerate(ltc):
                pred_labels[cluster] = i

            current_ami = adjusted_mutual_info_score(true_labels, pred_labels)

            if current_ami > ami:
                ami = current_ami

    else:
        pred_labels = np.zeros(n)

        for i, cluster in enumerate(ltc):
            pred_labels[cluster] = i

        ami = adjusted_mutual_info_score(true_labels, pred_labels)

    return ami


def dbscan(clustering_data: np.array, true_labels: np.array) -> np.float:
    """
    Perform standard k_means clustering and evaluate AMI against the ground-truth long-term clusters.

    Args:
        clustering_data (np.array): Data to be clustered.
        true_labels (np.array): Ground truth long-term clusters.
        max_iter (int): Maximum number of iterations for tkm.

    Returns:
        ami (float): Adjusted mutual info score.
    """
    t, m, n = clustering_data.shape
    k = len(np.unique(true_labels))

    weights = []
    for time in range(t):
        weights.append(DBSCAN(eps=0.5).fit_predict(clustering_data[time, :,:].T))

    weights = np.array(weights).T

    criteria_mat = similarity_matrix(weights)

    ltc = find_k_clusters(k=k, criteria_mat=criteria_mat, threshold_change=.05)

    if ltc is None:
        thresholds = np.linspace(.10, .90, 9)
        ami = 0

        for threshold in thresholds:
            ltc = find_long_term_clusters(similarity_threshold=threshold, criteria_mat=criteria_mat)

            pred_labels = np.zeros(n)

            for i, cluster in enumerate(ltc):
                pred_labels[cluster] = i

            current_ami = adjusted_mutual_info_score(true_labels, pred_labels)

            if current_ami > ami:
                ami = current_ami

    else:
        pred_labels = np.zeros(n)

        for i, cluster in enumerate(ltc):
            pred_labels[cluster] = i

        ami = adjusted_mutual_info_score(true_labels, pred_labels)

    return ami

