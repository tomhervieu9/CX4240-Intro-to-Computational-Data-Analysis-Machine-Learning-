
1.1 Programming - Explanation of code skeleton

In this programming assignment, you are going to apply the Bayes Classifier to handwritten digits classification
problem. Here, we use the binary 0/1 loss for binary classification (also referred to as error, or
accuracy throughout this instructions).

To ease your implementation, we provide a dataset file, usps 2cls.mat, that contains two categories
from USPS dataset. In this dataset, the first 256 columns are the features and the last column is their labels.
We also provide hw3.py that will (1) call function splitData to perform random split on data into
training and testing sets and (2) call your implementations in classifiers.py.
Function splitData is designed to perform random shuffle on data and split it into training set and testing
set given a predefined ratio and filename for a .mat file. You do not need to modify this function until you
wish to test your code with a different classification dataset (potentially with a different file extension; e.g.
.csv, .tsv, etc)

1.2 Programming - Your task
Your task is implementing the classifier by assuming the covariance matrix is (a) full, (b) diagonal, and
(c) spherical (Σ = cI). The functions you need to implement are in file classifiers.py and are named
modelFull, modelDiagonal and modelSpherical. The functions must contain code for (1) performing classification
on input sets using Bayes classifier and (2) evaluating accuracy of classification. Your code should
return this accuracy; for example, modelFull function takes in training and testing sets and returns two
values - accuracy on training and testing sets.

In hw3.py, call to your functions are repeated for 100 runs, for different portions for training/testing sets.
You can retrieve average accuracy over 100 runs by running python hw3.py on the command line. Hint -
try running python hw3.py as is; it’ll report 0 for average accuracy over 100 runs, as the function bodies in
classifier.py are empty.
