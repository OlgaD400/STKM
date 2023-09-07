import pandas as pd
import numpy as np
from distance_functions import s_journey
from tkm.graph_clustering_functions import STGKM, visualize_graph

final_voter_data = pd.read_csv('final_voter_data.csv')
fvd_0 = pd.read_csv('fvd_0')

voter_connectivity = np.load('roll_call_connectivity_2.npy')

distance_matrix = s_journey(voter_connectivity)

np.save('roll_call_distance.npy', distance_matrix)

distance_matrix = np.load('roll_call_distance.npy')

SUBSET = 100
subset_matrix = distance_matrix[:SUBSET]
t,n,_ = distance_matrix.shape
k = 3
stgkm = STGKM(distance_matrix = subset_matrix, penalty = 5, max_drift = 1, center_connectivity = 3, k = k, tie_breaker=False, iterations = 100)
penalized_distance = stgkm.penalize_distance()
# stgkm.first_kmeans(distance_matrix=penalized_distance)
stgkm.run_stgkm(method = 'proxy')



# for t in range(100):
#     clust_0_0_ids = np.where(stgkm.full_assignments[t*k] == 1)[0]
#     clust_1_0_ids = np.where(stgkm.full_assignments[k*t + 1]==1)[0]

#     print('Time', t)
#     print(fvd_0[fvd_0['legislator_id'].isin(clust_0_0_ids)]['party'].value_counts())
#     print(fvd_0[fvd_0['legislator_id'].isin(clust_1_0_ids)]['party'].value_counts())
#     print('\n\n')

# for t in range(100):
#     print('time', t)
#     for cluster in range(k):
#         ids = np.where(stgkm.full_assignments[t*k + cluster] == 1)[0]
#         members = fvd_0[fvd_0['legislator_id'].isin(ids)]['legislator'].to_list()

#         cluster_votes = final_voter_data[(final_voter_data['legislator'].isin(members)) & (final_voter_data['time'] == t)]['vote'].value_counts()
#         cluster_party = fvd_0[fvd_0['legislator_id'].isin(ids)]['party'].value_counts()

#         print('cluster', cluster, 'votes', cluster_votes, '\n\n party', cluster_party, '\n\n')


# ids = np.where(mems[0] == 1)[0]
# names = fvd_0[fvd_0['legislator_id'].isin(ids)]['legislator'].to_list()
# final_voter_data[(final_voter_data['time'] == 7) & (final_voter_data['legislator'].isin(names))]

#for cluster in range(k):
    # ids = np.where(stgkm.ltc == cluster)[0]
    # names = fvd_0[fvd_0['legislator_id'].isin(ids)]['legislator'].to_list()
    # final_voter_data[(final_voter_data['time'] == 7) & (final_voter_data['legislator'].isin(names))]['party'].value_counts()


# ids = np.where(stgkm.ltc == 2)[0]
# names = fvd_0[fvd_0['legislator_id'].isin(ids)]['legislator'].to_list()
# frac_no_vote = []
# for name in names:
#     legislator_df = final_voter_data[(final_voter_data['legislator']== name)]
#     frac_no_vote.append(len(legislator_df[legislator_df['vote'] == 'Not Voting'])/len(legislator_df))
# frac_no_vote.sort()
# print(frac_no_vote[::-1])
# plt.figure()
# y = stgkm.full_assignments[:,0]
# x = np.arange(len(y))
# plt.plot(x, y)

# timeslices = np.array([10,20,30,40,50,60,70,80,90])
# num_voters = 50
# labels = np.zeros((len(timeslices), num_voters))
# members = np.random.choice(432, num_voters)

# for index, time in enumerate(timeslices):
#     time_labels = np.argmax(stgkm.full_assignments[time*2: time*(2 + 1)], axis = 0)
#     labels[index] = time_labels[members]
                            
# # labels = stgkm.ltc[members]

# members = members.astype(int)
# visualize_graph(connectivity_matrix=voter_connectivity[timeslices][:,members][:,:, members], labels = labels)

# for time in final_voter_data['time'].unique():
#     timeslice = final_voter_data[final_voter_data['time']]
#     timeslice[timeslice['party'] == 'D']['vote']

#63, 70, 209
#How often do these guys vote along party lines 
#vs how often do they vote together

# ids = [63, 270, 209]
# members = fvd_0[fvd_0['legislator_id'].isin(ids)]['legislator'].to_list()
# member_vote_array = np.zeros((432, 100))

# fvd = final_voter_data.copy()
# fvd.replace(to_replace = 'Aye', value = 'Yea', inplace = True)
# fvd.replace(to_replace = 'No', value = 'Nay', inplace = True)

# dem_vote_history = []
# repub_vote_history = []
# for time in range(100):
#     dem_vote = fvd[(fvd['time'] == time) 
#                                 & (fvd['party'] == 'D')]['vote'].mode().values[0]
#     repub_vote = fvd[(fvd['time'] == time) 
#                                   & (fvd['party'] == 'R')]['vote'].mode().values[0]

#     dem_vote_history.append(dem_vote)
#     repub_vote_history.append(repub_vote)

#     for index, member in enumerate(legislators):
#         leg_vote = fvd[(fvd['time'] == time) 
#                                        & (fvd['legislator']== member)]['vote'].values[0]
    
#         if leg_vote == dem_vote:
#             member_vote_array[index, time] = 0 
#         elif leg_vote == repub_vote:
#             member_vote_array[index, time]=1
#         else:
#             member_vote_array[index, time]=2


# import random

# random_reps = random.sample(list(np.where(stgkm.ltc==2)[0]),3)
# for rep in random_reps:
#     label = fvd_0[fvd_0['legislator_id'] == rep]['last_name'].values[0]
#     plt.hist(sim_mat[rep], 
#             label=label, alpha = 0.7)
    
# plt.legend(loc='upper right')
# plt.title('Similarity Scores of Outlier Cluster')
# plt.savefig('Outliers.pdf', format = 'pdf')
# plt.show()
