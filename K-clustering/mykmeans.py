"""
    Your goal of this assignment is implementing your own K-means.

    Input:
         pixels: data set. Each row contains one data point. For image
         dataset, it contains 3 columns, each column corresponding to Red,
         Green, and Blue component.

         K: the number of desired clusters. Too high value of K may result in
         empty cluster error. Then, you need to reduce it.

    Output:
         assignment: the class assignment of each data point in pixels. The
         assignment should be 1, 2, 3, etc. For K = 5, for example, each cell
         of class should be either 1, 2, 3, 4, or 5. The output should be a
         column vector with size(pixels, 1) elements.

         centroid: the location of K centroids in your result. With images,
         each centroid corresponds to the representative color of each
         cluster. The output should be a matrix with size(pixels, 1) rows and
         3 columns. The range of values should be [0, 255].

     To illustrate, sklearn's kmeans function was used. Your job is to replace
     the call to sklearn's kemans function with your OWN implementation.

     You will get ZERO points for not modifying this script, or using other library
     to call k-means clustering.
"""
# from sklearn.cluster import KMeans
import numpy as np



def mykmeans(pixels, K):

    # ===== COMMENT BELOW LINES OUT AND IMPLEMENT YOUR OWN FUNCTION HERE =====
    # kmeans = KMeans(n_clusters=int(K), random_state=0).fit(pixels)
        #K- # of clusters
        #random_State fixes the seed and defaults to global numpy random number generator
        #.fit() - method of KMEANS that trains instances to cluster (takes in array-like or sparse matrix)- computes kmeans clustering

    # assignment = kmeans.labels_
    # centroid = kmeans.cluster_centers_
    # ========================================================================

    K = int(K)
    print pixels.shape
    centroidIndex = np.random.randint(pixels.shape[0], size = (1,K))
    centroid = pixels[centroidIndex,:]
    assignment = [0]*pixels.shape[0]

    # t = 0
    # while t in range(3):
    #Used basic while loop only to test code
    #Intended to use the following condition for convergence:
    #np.array_equal(np.sort(new_centroids, axis = 0), np.sort(centroids, axis = 0))
    for i in range(pixels.shape[0]):
		eucDist = [None]*K
		for j in range(K):
			eucDist[j] = np.sum(np.power((pixels[i] - centroid[0][j]),2))
		assignment[i] = np.argmin(eucDist)
    for j in range(K):
        classArray = []
        for x in range(pixels.shape[0]):
            if (assignment[x] == j):
            	classArray.append(pixels[x])
    	centroid[0][j] = np.mean(classArray, axis = 0)
	    # t = t + 1

    newestCentroid = np.empty((pixels.shape[0],1,3))
    for i in range(len(assignment)):
        newestCentroid[i][0] = centroid[0][assignment[i]]
    centroid = newestCentroid

    assignment = np.matrix(assignment)
    assignment = assignment.T
    return assignment, centroid