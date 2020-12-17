import numpy as np
import scipy.io as sio
import matplotlib
import matplotlib.pyplot as plt
from numpy.matlib import repmat
from sklearn.preprocessing import normalize
from scipy.io import loadmat

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
# viewcolumn(columnvector);
# VIEWCOLUMN Displays a 28 x 28 grayscale image stored in a column vector.
# Tim Marks 2002

def viewcolumn(columnvector):
    plt.imshow(columnvector.reshape([28, 28], order='F').T, cmap=plt.get_cmap('gray'))
    
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
# print(U.shape) #(784,784)

#printing eigen digits
# plt.figure()
# fig = plt.gcf()
# fig.suptitle("784 principal components of digit, 0-195", fontsize=14)
# for i in range(0,14):
    # for j in range(0,14):
        # plt.subplot(14,14,i*14+j+1)
        # plt.imshow(U[:,i*14+j].reshape([28, 28], order='F').T, cmap=plt.get_cmap('gray'))
# plt.show()

# plt.figure()
# fig = plt.gcf()
# fig.suptitle("784 principal components of digit, 196-391", fontsize=14)
# for i in range(0+14,14+14):
    # for j in range(0,14):
        # plt.subplot(14,14,(i-14)*14+j+1)
        # plt.imshow(U[:,i*14+j].reshape([28, 28], order='F').T, cmap=plt.get_cmap('gray'))
# plt.show()

# plt.figure()
# fig = plt.gcf()
# fig.suptitle("784 principal components of digit, 392-587", fontsize=14)
# for i in range(0+14+14,14+14+14):
    # for j in range(0,14):
        # plt.subplot(14,14,(i-14-14)*14+j+1)
        # plt.imshow(U[:,i*14+j].reshape([28, 28], order='F').T, cmap=plt.get_cmap('gray'))
# plt.show()

# plt.figure()
# fig = plt.gcf()
# fig.suptitle("784 principal components of digit, 588-783", fontsize=14)
# for i in range(0+14+14+14,14+14+14+14):
    # for j in range(0,14):
        # plt.subplot(14,14,(i-14-14-14)*14+j+1)
        # plt.imshow(U[:,i*14+j].reshape([28, 28], order='F').T, cmap=plt.get_cmap('gray'))
# plt.show()

digit1 = data_train[:,0]
digit1.shape = (784,1)

# plt.figure()
# viewcolumn(digit1)
# plt.show()

c = np.matmul(U.T, digit1 - mean_data_train)
# z = np.matmul(U, c) + mean_data_train

# plt.figure()
# viewcolumn(z)
# plt.show()

# print(U[:,0:99].shape) #(784, 99)
# z = np.matmul(U[:,0:99], c[0:99,:]) + mean_data_train

# plt.figure()
# viewcolumn(z)
# plt.show()

# print(c[0:20,:].shape) #(20,1)
z = np.matmul(U[:,0:20], c[0:20,:]) + mean_data_train

# plt.figure()
# viewcolumn(z)
# plt.show()

#makeing a 60000,20 matrix with components
train_data_20_P = np.zeros((60000,20)) #(60000, 20)
for i  in range(60000):
    digit = data_train[:,i]
    digit.shape = (784,1)
    c = np.matmul(U.T, digit - mean_data_train)
    train_data_20_P[i] = c[0:20,:].T
    
# C is 1 by N matrix in the format [c1 c2 ... cN]
# it indecates the grouping index 0 - K-1 for N points
# now C is set to be sutiable for train_data_20_P and train_data_20_D_min
# doing so is to record
C_for_return = np.zeros((1, train_data_20_P.shape[0]))

# record down each iteration's J_clust value
J_clust_list = []

clusters_count = 20
Kzs_for_return = np.zeros((clusters_count, train_data_20_P.shape[1]))

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
    runKMeans(train_data_20_P, clusters_count)
    np.savetxt('J_clust_list_K_20_P_run_' +str(i) +'.csv', J_clust_list, delimiter=',')
    np.savetxt('C_K_20_P_run_' +str(i) +'.csv', C_for_return, delimiter=',')
    np.savetxt('Kzs_K_20_P_run_' +str(i) +'.csv', Kzs_for_return, delimiter=',')