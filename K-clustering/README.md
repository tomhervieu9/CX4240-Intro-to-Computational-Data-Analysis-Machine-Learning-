You will be using PCA to perform dimensionality reduction on the given dataset (q4.csv). This dataset
contains vectorized grey scale photos of 62 students that were enrolled in this course in the past. The file
can be loaded as a matrix of 62 ’faces’. More specifically, there will be 4500 rows and 62 columns where
each column corresponds to one face. You are to use Principal Component Analysis to perform Image
Compression.
• Submit a plot of the Eigen values in ascending order (Visualize the increase of Eigen values across all
Eigen vectors).
• Select a cut off to choose the top k eigen-vectors (or eigenfaces) based on the graph. Discuss the
reasoning for choosing this cut off.
• For your chosen eigen faces, calculate the reconstruction error (Squared distance from original image,
and reconstructed image) for the first two images in the dataset.
• Vary the number of eigen faces to view the differences in reconstruction error and in the quality of the
image. Display images of your chosen n eigen faces, and attach two images to your submission.

#Note: Eigenvalue are off because data was not normalized. Never forget to normalize data in future algorithm implementations!
