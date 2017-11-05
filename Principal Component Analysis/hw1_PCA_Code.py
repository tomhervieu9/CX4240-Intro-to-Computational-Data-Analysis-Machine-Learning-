 import numpy as np
import matplotlib.pyplot as plt

#load matrix
matrix = np.genfromtxt('C:\Users\Tom\Documents\CX4240\HW1\HW1_Thomas_Hervieu\q4.csv', dtype ='float', delimiter = ',')
matrix = matrix.T


#PCA
# Subtracting the mean of the dataset
m = matrix.shape[1]
mu = matrix.sum(axis=1) / m
mu  = mu.reshape((mu.shape[0],1))
xc = matrix - mu

# Finding the covariance 
C = xc.dot(xc.T) / m

#Show Eigenvalues
eigenvalues, eigenvectors = np.linalg.eigh(C)
# eigenvectors = eigenvectors/np.linalg.norm(eigenvectors)
plt.figure(1)
plt.plot(eigenvalues)
plt.xlabel('Number of Eigenvalues')
plt.ylabel('Eigenvalues')
plt.title('Eigenvalues in Ascending Order')
plt.show()

# Finding top k pricipal component(eigen vector of the covariance)
k = 20

sortidx = eigenvalues.argsort()[::-1]
eigenvalues = eigenvalues[sortidx][0:k]
eigenvectors = eigenvectors[:,sortidx][:,0:k]
eigenvalues = np.diag(eigenvalues)


# # python obtains the opposite eigenvector
eigenvectors[:, 0] = -eigenvectors[:, 0]

# # # projecting the 62x4500 data on the k eigenvectors
# dim1 = eigenvectors[:, 0].T.dot(xc) / np.sqrt(eigenvalues[0,0])
# dim2 = eigenvectors[:, 1].T.dot(xc) / np.sqrt(eigenvalues[1,1])


#Reconstructing the Matrix
eigenvectors = eigenvectors.T

recMatrix = np.zeros((1,4500))
for i in range(0,k):
	sum = eigenvectors[i].dot(xc)
	recMatrix += sum
mu = mu.sum(axis=0) / 62
recMatrix += mu
recMatrix = np.reshape(recMatrix, (75, 60), order='F')

#Displaying the Reconstructed Matrix
plt.imshow(recMatrix, cmap='gray')
plt.show()


# # Reconstruction Error
matrix1 = np.reshape(matrix[0], (75,60), order='F')
error1 = np.linalg.norm(matrix1 - recMatrix)
print error1

matrix2 = np.reshape(matrix[1], (75,60), order='F')
error2 = np.linalg.norm(matrix2 - recMatrix)
print error2

print error1, error2