1 Programming: Image compression using K-means [40 pts]

In this programming assignment, you are going to apply clustering algorithms for image compression. Before
starting this assignment, we strongly recommend reading PRML Section 9.1.1, page 428 – 430.
To ease your implementation, we provide a skeleton code containing image processing part. hw2.py is
designed to read an RGB bitmap image file, then cluster pixels with the given number of clusters K. It shows
converted image only using K colors, each of them with the representative color of centroid. To see what
it looks like, you are encouraged to run python hw2.py beach.bmp 5 or python hw2.py football.bmp 3,
for example.

Your task is implementing the clustering parts with K-means. We learned and demonstrated K-means
in class, so you may start from the sample code we distributed.
The file you need to edit is mykmeans.py, provided with this homework. In the files, you can see it calls
function sklearn.cluster.KMeans initially. Sklearn is a library with pre-implemented KMeans function.
Comment this line out, and implement your own in the files. You would expect to see similar result with
your implementation of K-means, instead of sklearn.cluster.KMeans function.
Formatting information is here:

Input
• pixels: data set. Each row contains one data point. For image dataset, it contains 3 columns, each
column corresponding to Red, Green, and Blue component.
• K: the number of desired clusters. Too high value of K may result in empty cluster error. Then, you
need to reduce it.
1Christopher M. Bishop, Pattern Recognition and Machine Learning, 2006, Springer.
1

Output
• class: the class assignment of each data point in pixels. The assignment should be 1, 2, 3, etc. For
K = 5, for example, each cell of class should be either 1, 2, 3, 4, or 5. The output should be a column
vector with size(pixels, 1) elements.
• centroid: the location of K centroids in your result. With images, each centroid corresponds to the
representative color of each cluster. The output should be a matrix with K rows and 3 columns. The
range of values should be [0, 255].
