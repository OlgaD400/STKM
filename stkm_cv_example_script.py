"""Example script for running stkm for cv problems."""
from stkm.STKM import STKM
import numpy as np
import matplotlib.pyplot as plt
from cv_stkm import generate_input_data, return_predicted_masked_image


### toy data
# timesteps = 10
# dimension = 3
# num_points = 100
# num_clusters = 2

# data = np.random.rand(timesteps, dimension, num_points)
# weights = np.random.rand(timesteps, num_points, num_clusters)
# centers = np.random.rand(timesteps, dimension, num_clusters)

# stkm = STKM(data=data)
# stkm.perform_clustering(num_clusters=2, method="L1")

# plt.figure()
# plt.title("Error")
# plt.plot(tkm.err_hist)

# plt.figure()
# plt.title("Objective")
# plt.plot(tkm.obj_hist)

#######################################################################
dir = "cv_data/train/JPEGImages/0f9683715b/"  # 1a6c0fbd1e/"
image_paths, input_data = generate_input_data(image_directory=dir)

stkm = STKM(input_data)
stkm.perform_clustering(num_clusters=2, lam=0.8, max_iter=500, method="L1", gamma=1e-3)

# return_predicted_masked_image(image_paths=image_paths, index=10, weights=stkm.weights)
