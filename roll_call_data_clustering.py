import pandas as pd
import numpy as np
from distance_functions import s_journey
from tkm.graph_clustering_functions import STGKM 

final_voter_data = pd.read_csv('final_voter_data.csv')
fvd_0 = pd.read_csv('fvd_0')

voter_connectivity = np.load('roll_call_connectivity.npy')

# distance_matrix = s_journey(voter_connectivity)

# np.save('roll_call_distance.npy', distance_matrix)

distance_matrix = np.load('roll_call_distance.npy')

subset_matrix = distance_matrix[:100]
stgkm = STGKM(distance_matrix = subset_matrix, penalty = 5, max_drift = 1, k = 2, tie_breaker=False, iter = 100)
penalized_distance = stgkm.penalize_distance()
# stgkm.first_kmeans(distance_matrix=penalized_distance)
stgkm.run_stgkm(method = 'proxy')

# for t in range(100):
#     clust_0_0_ids = np.where(stgkm.full_assignments[t*2] == 1)[0]
#     clust_1_0_ids = np.where(stgkm.full_assignments[2*t + 1]==1)[0]

#     print('Time', t)
#     print(fvd_0[fvd_0['legislator_id'].isin(clust_0_0_ids)]['party'].value_counts())
#     print(fvd_0[fvd_0['legislator_id'].isin(clust_1_0_ids)]['party'].value_counts())
#     print('\n\n')


# ids = np.where(mems[0] == 1)[0]
# names = fvd_0[fvd_0['legislator_id'].isin(ids)]['legislator'].to_list()
# final_voter_data[(final_voter_data['time'] == 7) & (final_voter_data['legislator'].isin(names))]

# ids = np.where(stgkm.ltc == 1)[0]
# names = fvd_0[fvd_0['legislator_id'].isin(ids)]['legislator'].to_list()
# final_voter_data[(final_voter_data['time'] == 7) & (final_voter_data['legislator'].isin(names))]['party].value_counts()