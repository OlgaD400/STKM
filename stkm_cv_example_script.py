
from stkm.STKM import STKM
import numpy as np 
import matplotlib.pyplot as plt
from cv_stkm import generate_input_data


### toy data
timesteps = 10
dimension = 3
num_points = 100
num_clusters = 2

data = np.random.rand(timesteps,dimension,num_points)
weights = np.random.rand(timesteps,num_points,num_clusters)
centers = np.random.rand(timesteps,dimension,num_clusters)

tkm = STKM(data = data)
tkm.perform_clustering(num_clusters = 2, method = 'L1')

plt.figure()
plt.title('Error')
plt.plot(tkm.err_hist)

plt.figure()
plt.title('Objective')
plt.plot(tkm.obj_hist)

# dir = 'cv_data/train/JPEGImages/0b34ec1d55/' #0ae1ff65a5/'
# image_paths, input_data = generate_input_data(image_directory=dir)

# tkm = TKM(input_data)
# tkm.perform_clustering_cosine(num_clusters = 2, lam = .8, max_iter = 1000)


# if __name__ == '__main__':
#     cProfile.run('main()')