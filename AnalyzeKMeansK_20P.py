import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.preprocessing import normalize

X = loadmat("mnist.mat")
data_train = X['trainX'] # Training set digits (60000, 784)
data_train_label = X['trainY'] # Training set labels (1, 60000)
data_test = X['testX']
data_test_label = X['testY']

# %%
# [Vsort,Dsort] = eigsort(V, eigvals)
#
# Sorts a matrix eigenvectors and a array of eigenvalues in order 
# of eigenvalue size, largest eigenvalue first and smallest eigenvalue
# last.
#
# Example usage:
# di, V = np.linarg.eig(L)
# Vnew, Dnew = eigsort(V, di)
#
# Tim Marks 2002

# %%
def eigsort(V, eigvals):
    
    # Sort the eigenvalues from largest to smallest. Store the sorted
    # eigenvalues in the column vector lambd.
    lohival = np.sort(eigvals)
    lohiindex = np.argsort(eigvals)
    lambd = np.flip(lohival)
    index = np.flip(lohiindex)
    Dsort = np.diag(lambd)
    
    # Sort eigenvectors to correspond to the ordered eigenvalues. Store sorted
    # eigenvectors as columns of the matrix vsort.
    M = np.size(lambd)
    Vsort = np.zeros((M, M))
    for i in range(M):
        Vsort[:,i] = V[:,index[i]]
    return Vsort, Dsort

# %%
# normc(M) normalizes the columns of M to a length of 1.

def normc(Mat):
    return normalize(Mat, norm='l2', axis=0)

data_train = data_train.T
# plt.figure()
# viewcolumn(data_train[:,0])
# plt.show()
# print(data_train.shape) #(784, 60000)
mean_data_train = np.mean(data_train, axis=1, keepdims=True)
# print(mean_data_train.shape) #(784,1)

mean_data_train_matrix = np.zeros((784, 60000))
for i in range(784):
    for j in range(60000):
        mean_data_train_matrix[i][j] = mean_data_train[i][0]
        

A = data_train - mean_data_train_matrix
# print(A.shape)
eigvals, V = np.linalg.eig(np.matmul(A, A.T))
U, D = eigsort(V, eigvals)
U = normc(U)

#makeing a 60000,20 matrix with components
train_data_20_P = np.zeros((60000,20)) #(60000, 20)
for i  in range(60000):
    digit = data_train[:,i]
    digit.shape = (784,1)
    c = np.matmul(U.T, digit - mean_data_train)
    train_data_20_P[i] = c[0:20,:].T

data_train = data_train.T
#=============================================================================================================================
# this 10 th run has 100 recorded J_clust value 
# every 10 is for a single run with 10 iterations
J_clust_list = np.loadtxt('J_clust_list_K_20_P_run_9.csv', delimiter=',')
# print(J_clust_list.shape) # (10,)
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
        
print('for K = 20, total of 10 runs, min J_clust run is: ' + str(min_J_clust_index) + ' with J_clust = ' + str(min_J_clust), end = '')
print(' max J_clust run is: ' + str(max_J_clust_index) + ' with J_clust = ' + str(max_J_clust))

# K = 20, there are 10 runs, each run has 10 iterations (which is a decision)
k_20_min_J_clust_run_J_clust_recordings = []
k_20_max_J_clust_run_J_clust_recordings = []

# load J_clust recordings for max and min runs
for i in range(min_J_clust_index * 10, min_J_clust_index * 10 + 10):
    k_20_min_J_clust_run_J_clust_recordings.append(J_clust_list[i])
# print(k_20_min_J_clust_run_J_clust_recordings)    
    
for i in range(max_J_clust_index * 10, max_J_clust_index * 10 + 10):
    k_20_max_J_clust_run_J_clust_recordings.append(J_clust_list[i])
# print(k_20_max_J_clust_run_J_clust_recordings) 

# plotting the J_clust in each of the 10 iterations (remember k means algorithm is set to iterate 10 times)
plt.figure()
plt.title("plotting J_clust at each iteration, K = 20, max and min cases")
plt.plot([0,1,2,3,4,5,6,7,8,9], k_20_min_J_clust_run_J_clust_recordings)
plt.plot([0,1,2,3,4,5,6,7,8,9], k_20_max_J_clust_run_J_clust_recordings)
plt.show()

# load C grouping for max and min runs
k_20_min_J_clust_run_C_recordings = np.loadtxt('C_K_20_P_run_' + str(min_J_clust_index) + '.csv', delimiter=',')
k_20_max_J_clust_run_C_recordings = np.loadtxt('C_K_20_P_run_' + str(max_J_clust_index) + '.csv', delimiter=',')
# print(k_20_min_J_clust_run_C_recordings.shape) # (60000,)
# print(k_20_max_J_clust_run_C_recordings.shape) # (60000,)

# load K Z vectors (the centroid) grouping for max and min runs
k_20_min_J_clust_run_Z_vectors = np.loadtxt('Kzs_K_20_P_run_' + str(min_J_clust_index) + '.csv', delimiter=',')
k_20_max_J_clust_run_Z_vectors = np.loadtxt('Kzs_K_20_P_run_' + str(max_J_clust_index) + '.csv', delimiter=',')
# print(k_20_min_J_clust_run_Z_vectors.shape) # (20, 20)
# print(k_20_max_J_clust_run_Z_vectors.shape) # (20, 20)

# we have 20 Z's 
# now figuring out 10 closest points to each of these Z's
# we use C grouping info
# in C (60000,), each of the point is labeled into 0,...,19
# to figure out 10 closest point to ith Z, we need to compute and compare all the points labeled as i the norm to Zi
# then we choose top 10 shortest ones
min_closest_points = [] # eventually will have 200 pairs
max_closest_points = [] # eventually will have 200 pairs

for i in range(20):
    min_norms_list_for_Zi = []
    max_norms_list_for_Zi = []
    for j in range(k_20_min_J_clust_run_C_recordings.shape[0]):
        if(k_20_min_J_clust_run_C_recordings[j] == i):
            min_norms_list_for_Zi.append((np.linalg.norm(k_20_min_J_clust_run_Z_vectors[i,:]-train_data_20_P[j,:]),j))
    for j in range(k_20_max_J_clust_run_C_recordings.shape[0]):
        if(k_20_max_J_clust_run_C_recordings[j] == i):
            max_norms_list_for_Zi.append((np.linalg.norm(k_20_max_J_clust_run_Z_vectors[i,:]-train_data_20_P[j,:]),j))
    min_norms_list_for_Zi.sort(key=lambda elem: elem[0])
    max_norms_list_for_Zi.sort(key=lambda elem: elem[0])
    for j in range(10):
        min_closest_points.append(min_norms_list_for_Zi[j])
        max_closest_points.append(max_norms_list_for_Zi[j])
        
plt.figure()
fig = plt.gcf()
fig.suptitle("20 by 10, i'th row is 10 points is closest 10 points to Zi, then from the index going back to the image, this is the min run", fontsize=14)
for i in range(20):
    for j in range(10):
        plt.subplot(20,10,i*10+j+1)
        plt.imshow(data_train[min_closest_points[i*10+j][1]].reshape(28,28),cmap='binary')
plt.show()

plt.figure()
fig = plt.gcf()
fig.suptitle("20 by 10, i'th row is 10 points is closest 10 points to Zi, then from the index going back to the image, this is the max run", fontsize=14)
for i in range(20):
    for j in range(10):
        plt.subplot(20,10,i*10+j+1)
        plt.imshow(data_train[max_closest_points[i*10+j][1]].reshape(28,28),cmap='binary')
plt.show()