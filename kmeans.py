import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

X = loadmat("mnist.mat")
data_train = X['trainX'] # Training set digits (60000, 784)
data_train_label = X['trainY'] # Training set labels (1, 60000)
data_test = X['testX']
data_test_label = X['testY']

# C is 1 by N matrix in the format [c1 c2 ... cN]
# it indecates the grouping index 0 - K-1 for N points
# now C is set to be sutiable for data_train
# comment out this line 16 and uncomment line 133 for using kmeans for general input
# doing so is to record
C_for_return = np.zeros((1, data_train.shape[0]))

# record down each iteration's J_clust value
J_clust_list = []

clusters_count = 5
Kzs_for_return = np.zeros((clusters_count, data_train.shape[1]))

def plotCurrent(X, Rnk, Kmus):
    N, D = X.shape
    K = Kmus.shape[0]

    InitColorMat = np.array([[1, 0, 0],
                             [0, 1, 0],
                             [0, 0, 1],
                             [0, 0, 0],
                             [1, 1, 0],
                             [1, 0, 1],
                             [0, 1, 1]])

    KColorMat = InitColorMat[0:K,:]

    colorVec = np.dot(Rnk, KColorMat)
    muColorVec = np.dot(np.eye(K), KColorMat)
    plt.scatter(X[:,0], X[:,1], c=colorVec)

    plt.scatter(Kmus[:,0], Kmus[:,1], s=200, c=muColorVec, marker='d')
    plt.axis('equal')
    plt.show()

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
    runKMeans(data_train, clusters_count)
    np.savetxt('J_clust_list_K_5_run_' +str(i) +'.csv', J_clust_list, delimiter=',')
    np.savetxt('C_K_5_run_' +str(i) +'.csv', C_for_return, delimiter=',')
    np.savetxt('Kzs_K_5_run_' +str(i) +'.csv', Kzs_for_return, delimiter=',')
# runKMeans(data_train, clusters_count)
# J_clust values in each iteration
# print(J_clust_list)
# C 
# print(C_for_return)
# plotting z vectors
# plt.figure()
# for i in range(clusters_count):
    # plt.subplot(4,5,i+1)
    # plt.imshow(Kzs_for_return[i,:].reshape(28,28),cmap='binary')
# plt.show()