import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

X = loadmat("mnist.mat")
data_train = X['trainX'] # Training set digits (60000, 784)
data_train_label = X['trainY'] # Training set labels (1, 60000)
data_test = X['testX']
data_test_label = X['testY']

#=============================================================================================================================
# this 30 th run has 300 recorded J_clust value 
# every 10 is for a single run with 10 iterations
J_clust_list = np.loadtxt('J_clust_list_K_20_run_29.csv', delimiter=',')
# print(J_clust_list.shape) # (300,)
final_J_clust_for_each_run = []
for i in range(30):
    final_J_clust_for_each_run.append(J_clust_list[i*10 + 9])


min_J_clust = final_J_clust_for_each_run[0]
min_J_clust_index = 0
max_J_clust = final_J_clust_for_each_run[0]
max_J_clust_index = 0
for i in range(30):
    if(final_J_clust_for_each_run[i] < min_J_clust):
        min_J_clust = final_J_clust_for_each_run[i]
        min_J_clust_index = i
    if(final_J_clust_for_each_run[i] > max_J_clust):
        max_J_clust = final_J_clust_for_each_run[i]
        max_J_clust_index = i
        
# print('for K = 20, total of 30 runs, min J_clust run is: ' + str(min_J_clust_index) + ' with J_clust = ' + str(min_J_clust), end = '')
# print(' max J_clust run is: ' + str(max_J_clust_index) + ' with J_clust = ' + str(max_J_clust))


# K = 20, there are 30 runs, each run has 10 iterations (which is a decision)
k_20_min_J_clust_run_J_clust_recordings = []
k_20_max_J_clust_run_J_clust_recordings = []

# load J_clust recordings for max and min runs
for i in range(min_J_clust_index * 10, min_J_clust_index * 10 + 10):
    k_20_min_J_clust_run_J_clust_recordings.append(J_clust_list[i])
# print(k_20_min_J_clust_run_J_clust_recordings)    
    
for i in range(max_J_clust_index * 10, max_J_clust_index * 10 + 10):
    k_20_max_J_clust_run_J_clust_recordings.append(J_clust_list[i])
# print(k_20_max_J_clust_run_J_clust_recordings) 

# load C grouping for max and min runs
k_20_min_J_clust_run_C_recordings = np.loadtxt('C_K_20_run_' + str(min_J_clust_index) + '.csv', delimiter=',')
k_20_max_J_clust_run_C_recordings = np.loadtxt('C_K_20_run_' + str(max_J_clust_index) + '.csv', delimiter=',')
# print(k_20_min_J_clust_run_C_recordings.shape) # (60000,)
# print(k_20_max_J_clust_run_C_recordings.shape) # (60000,)

# load K Z vectors (the centroid) grouping for max and min runs
k_20_min_J_clust_run_Z_vectors = np.loadtxt('Kzs_K_20_run_' + str(min_J_clust_index) + '.csv', delimiter=',')
k_20_max_J_clust_run_Z_vectors = np.loadtxt('Kzs_K_20_run_' + str(max_J_clust_index) + '.csv', delimiter=',')
# print(k_20_min_J_clust_run_Z_vectors.shape) # (20, 784)
# print(k_20_max_J_clust_run_Z_vectors.shape) # (20, 784)

train_data_20_D_min = np.zeros((data_train.shape[0],20)) #(60000, 20)
for i in range(data_train.shape[0]):
    for j in range(20):
        train_data_20_D_min[i][j] = np.linalg.norm(k_20_min_J_clust_run_Z_vectors[j,:]-data_train[i,:])
        
# C is 1 by N matrix in the format [c1 c2 ... cN]
# it indecates the grouping index 0 - K-1 for N points
# now C is set to be sutiable for train_data_20_D_max and train_data_20_D_min
# doing so is to record
C_for_return = np.zeros((1, train_data_20_D_max.shape[0]))

# record down each iteration's J_clust value
J_clust_list = []

clusters_count = 20
Kzs_for_return = np.zeros((clusters_count, train_data_20_D_max.shape[1]))

# input: list_of_N_vectors N by D matrix, Kzs K by D matrix
# output: sqDmat N by K matrix, a squared distance matrix where the n,k entry
# contains the squared distance from the nth data vector to the kth z vector
def calcSqDistances(list_of_N_vectors, Kzs):
    sqDmat = np.zeros((list_of_N_vectors.shape[0], Kzs.shape[0])) # N by K zero matrix
    for i in range(0, sqDmat.shape[0]): #iterate through all the rows of sqDmat
        for j in range(0, sqDmat.shape[1]): #iterate through all the cols of sqDmat
            sqDmat[i][j] = pow(np.linalg.norm(list_of_N_vectors[i,:]-Kzs[j,:]),2)
            
            
            
    return sqDmat
    
def determineRnk(sqDmat):
    Rnk = np.zeros((sqDmat.shape[0], sqDmat.shape[1])) # N by K zero matrix
    for i in range(0, sqDmat.shape[0]): #iterate through all the rows of sqDmat
        Rnk[i][np.argmin(sqDmat[i])] = 1 # minIndex = 0 # min index tracker
        # minSQDistance = sqDmat[i][0] # min squared distance tracker
        # for j in range(0, sqDmat.shape[1]): #iterate through all the cols of sqDmat
            # if(sqDmat[i][j] < minSQDistance): #find a distance that is less
                # minIndex = j #update the index
                # minSQDistance = sqDmat[i][j] #update the min squared distance tracker
        # Rnk[i][minIndex] = 1
    return Rnk
    
def recalcZs(list_of_N_vectors, Rnk):
    Kzs = np.zeros((Rnk.shape[1], list_of_N_vectors.shape[1])) #K by D zero matrix
    for j in range(0, Rnk.shape[1]): #iterate through the cols of Rnk, K cols
        count = 0 #track how many points are closest to this jth k
        sumOfAllPointsInThisCategory = np.zeros((1, list_of_N_vectors.shape[1])) #initialize to zero vector
        for i in range(0, Rnk.shape[0]): #iterate through the rows of Rnk, N
            if(Rnk[i][j] == 1):
                count += 1
                sumOfAllPointsInThisCategory += list_of_N_vectors[i]
        Kzs[j] = sumOfAllPointsInThisCategory/count
    #print(Kzs)
    return Kzs # K by D, each Z vector in rows, K rows in total
    
# input: a list of N data vectors, number of clusters K 
# output: a list of N group assignment indices (c1, c2, ... ,cN), a list of K group representative vectors (z1, ... ,zK), value of J_clust after each iteration
def runKMeans(list_of_N_vectors, K):
    #determine and store data set information
    N, D = list_of_N_vectors.shape # N is row number, D is column number
    
    #allocate space for the K z vectors
    Kzs = np.zeros((K, D)) # K by D matrix
    
    #initialize cluster centers by randomly picking points from the data
    #randomly assignKdata points as the K groups representatives
    rand_inds = np.random.permutation(N) # random sequence
    Kzs = list_of_N_vectors[rand_inds[0:K],:] # K by D matrix
    
    # for i in range(K):
        # plt.figure()
        # plt.imshow(Kzs[i,:].reshape(28,28),cmap='binary')
        # plt.show()
        
    # plt.figure()
    # for i in range(K):
        # plt.subplot(4,5,i+1)
        # plt.imshow(Kzs[i,:].reshape(28,28),cmap='binary')
    # plt.show()
    
    #specify the maximum number of iterations to allow
    maxiters = 10
    
    for iter in range(maxiters):
        #assign each data vector to closest z vector
        #do this by first calculating a squared distance matrix where the n,k entry
        #contains the squared distance from the nth data vector to the kth z vector

        #sqDmat will be an N-by-K matrix with the n,k entry as specfied above
        sqDmat = calcSqDistances(list_of_N_vectors, Kzs)
        #print(sqDmat)
        
        #given the matrix of squared distances, determine the closest cluster
        #center for each data vector

        #R is the "responsibility" matrix
        #R will be an N-by-K matrix of binary values whose n,k entry is set as
        #per Bishop (9.2)
        #Specifically, the n,k entry is 1 if point n is closest to cluster k,
        #and is 0 otherwise
        Rnk = determineRnk(sqDmat)
        #print(Rnk)
        
        # C is 1 by N matrix in the format [c1 c2 ... cN]
        # it indecates the grouping index 0 - K-1 for N points
        C = np.zeros((1, N))
        for i in range(Rnk.shape[0]):
            for j in range(Rnk.shape[1]):
                if(Rnk[i][j] == 1):
                    C[0][i] = j
        if(iter == maxiters - 1):
            for i in range(C.shape[1]):
                C_for_return[0][i] = C[0][i]

        KzsOld = Kzs
        # plotCurrent(list_of_N_vectors, Rnk, Kzs)
        # time.sleep(1)

        #recalculate mu values based on cluster assignments
        # K by D matrix with zi in each row
        Kzs = recalcZs(list_of_N_vectors, Rnk)
            
        if(iter == maxiters - 1):
            for i in range(Kzs.shape[0]):
                for j in range(Kzs.shape[1]):
                    Kzs_for_return[i][j] = Kzs[i][j]
            
        # plt.figure()
        # for i in range(K):
            # plt.subplot(4,5,i+1)
            # plt.imshow(Kzs[i,:].reshape(28,28),cmap='binary')
        # plt.show()

        # calculate J_clust
        J_clust = 0
        for i in range(Kzs.shape[0]): # iterate through K z vectors
            J_i = 0
            for j in range(C.shape[1]):
                if(C[0][j] == i):
                   J_i +=  pow(np.linalg.norm(list_of_N_vectors[j,:]-Kzs[i,:]),2)
            J_i = J_i/(C.shape[1])
            J_clust += J_i
        print(J_clust)
        J_clust_list.append(J_clust)
    return J_clust
    
for i in range(10):
    runKMeans(train_data_20_D_min, clusters_count)
    np.savetxt('J_clust_list_K_20_D_run_' +str(i) +'.csv', J_clust_list, delimiter=',')
    np.savetxt('C_K_20_D_run_' +str(i) +'.csv', C_for_return, delimiter=',')
    np.savetxt('Kzs_K_20_D_run_' +str(i) +'.csv', Kzs_for_return, delimiter=',')