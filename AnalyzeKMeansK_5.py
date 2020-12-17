import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

X = loadmat("mnist.mat")
data_train = X['trainX'] # Training set digits (60000, 784)
data_train_label = X['trainY'] # Training set labels (1, 60000)
data_test = X['testX']
data_test_label = X['testY']

#=============================================================================================================================
# this 10 th run has 100 recorded J_clust value 
# every 10 is for a single run with 10 iterations
J_clust_list = np.loadtxt('J_clust_list_K_5_run_9.csv', delimiter=',')
# print(J_clust_list.shape) # (100,)
final_J_clust_for_each_run = []
for i in range(10):
    final_J_clust_for_each_run.append(J_clust_list[i*10 + 9])


min_J_clust = final_J_clust_for_each_run[0]
min_J_clust_index = 0
max_J_clust = final_J_clust_for_each_run[0]
max_J_clust_index = 0
for i in range(10):
    if(final_J_clust_for_each_run[i] < min_J_clust):
        min_J_clust = final_J_clust_for_each_run[i]
        min_J_clust_index = i
    if(final_J_clust_for_each_run[i] > max_J_clust):
        max_J_clust = final_J_clust_for_each_run[i]
        max_J_clust_index = i
        
print('for K = 5, total of 10 runs, min J_clust run is: ' + str(min_J_clust_index) + ' with J_clust = ' + str(min_J_clust), end = '')
print(' max J_clust run is: ' + str(max_J_clust_index) + ' with J_clust = ' + str(max_J_clust))


# K = 5, there are 10 runs, each run has 10 iterations (which is a decision)
k_5_min_J_clust_run_J_clust_recordings = []
k_5_max_J_clust_run_J_clust_recordings = []

# load J_clust recordings for max and min runs
for i in range(min_J_clust_index * 10, min_J_clust_index * 10 + 10):
    k_5_min_J_clust_run_J_clust_recordings.append(J_clust_list[i])
# print(k_5_min_J_clust_run_J_clust_recordings)    
    
for i in range(max_J_clust_index * 10, max_J_clust_index * 10 + 10):
    k_5_max_J_clust_run_J_clust_recordings.append(J_clust_list[i])
# print(k_5_max_J_clust_run_J_clust_recordings) 

# plotting the J_clust in each of the 10 iterations (remember k means algorithm is set to iterate 10 times)
plt.figure()
plt.title("plotting J_clust at each iteration, K = 5, max and min cases")
plt.plot([0,1,2,3,4,5,6,7,8,9], k_5_min_J_clust_run_J_clust_recordings)
plt.plot([0,1,2,3,4,5,6,7,8,9], k_5_max_J_clust_run_J_clust_recordings)
plt.show()

# load C grouping for max and min runs
k_5_min_J_clust_run_C_recordings = np.loadtxt('C_K_5_run_' + str(min_J_clust_index) + '.csv', delimiter=',')
k_5_max_J_clust_run_C_recordings = np.loadtxt('C_K_5_run_' + str(max_J_clust_index) + '.csv', delimiter=',')
# print(k_5_min_J_clust_run_C_recordings.shape) # (60000,)
# print(k_5_max_J_clust_run_C_recordings.shape) # (60000,)

# load K Z vectors (the centroid) grouping for max and min runs
k_5_min_J_clust_run_Z_vectors = np.loadtxt('Kzs_K_5_run_' + str(min_J_clust_index) + '.csv', delimiter=',')
k_5_max_J_clust_run_Z_vectors = np.loadtxt('Kzs_K_5_run_' + str(max_J_clust_index) + '.csv', delimiter=',')
# print(k_5_min_J_clust_run_Z_vectors.shape) # (5, 784)
# print(k_5_max_J_clust_run_Z_vectors.shape) # (5, 784)

plt.figure()
fig = plt.gcf()
fig.suptitle("K = 5, z vectors in the min J_clust run out of 10 runs", fontsize=14)
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(k_5_min_J_clust_run_Z_vectors[i,:].reshape(28,28),cmap='binary')
plt.show()

plt.figure()
fig = plt.gcf()
fig.suptitle("K = 5, z vectors in the max J_clust run out of 30 runs", fontsize=14)
for i in range(5):
    plt.subplot(1,5,i+1)
    plt.imshow(k_5_max_J_clust_run_Z_vectors[i,:].reshape(28,28),cmap='binary')
plt.show()

# we have 5 Z's 
# now figuring out 10 closest points to each of these Z's
# we use C grouping info
# in C (60000,), each of the point is labeled into 0,...,4
# to figure out 10 closest point to ith Z, we need to compute and compare all the points labeled as i the norm to Zi
# then we choose top 10 shortest ones
min_closest_points = [] # eventually will have 200 pairs
max_closest_points = [] # eventually will have 200 pairs

for i in range(5):
    min_norms_list_for_Zi = []
    max_norms_list_for_Zi = []
    for j in range(k_5_min_J_clust_run_C_recordings.shape[0]):
        if(k_5_min_J_clust_run_C_recordings[j] == i):
            min_norms_list_for_Zi.append((np.linalg.norm(k_5_min_J_clust_run_Z_vectors[i,:]-data_train[j,:]),j))
    for j in range(k_5_max_J_clust_run_C_recordings.shape[0]):
        if(k_5_max_J_clust_run_C_recordings[j] == i):
            max_norms_list_for_Zi.append((np.linalg.norm(k_5_max_J_clust_run_Z_vectors[i,:]-data_train[j,:]),j))
    min_norms_list_for_Zi.sort(key=lambda elem: elem[0])
    max_norms_list_for_Zi.sort(key=lambda elem: elem[0])
    for j in range(10):
        min_closest_points.append(min_norms_list_for_Zi[j])
        max_closest_points.append(max_norms_list_for_Zi[j])
        
# print(len(min_closest_points)) # 50, every 10 is 10 closest points, 5 sets of 10 points, each element is a pair, pair[1] is the index in data_train
# print(len(max_closest_points)) # 50, every 10 is 10 closest points, 5 sets of 10 points, each element is a pair, pair[1] is the index in data_train

plt.figure()
fig = plt.gcf()
fig.suptitle("5 by 10, i'th row is 10 points is closest 10 points to Zi, this is the min run", fontsize=14)
for i in range(5):
    for j in range(10):
        plt.subplot(5,10,i*10+j+1)
        plt.imshow(data_train[min_closest_points[i*10+j][1]].reshape(28,28),cmap='binary')
plt.show()

plt.figure()
fig = plt.gcf()
fig.suptitle("5 by 10, i'th row is 10 points is closest 10 points to Zi, this is the max run", fontsize=14)
for i in range(5):
    for j in range(10):
        plt.subplot(5,10,i*10+j+1)
        plt.imshow(data_train[max_closest_points[i*10+j][1]].reshape(28,28),cmap='binary')
plt.show()
#============================================================================================================================= 