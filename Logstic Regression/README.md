
1 Multi-class Logistic Regression

(a) Derive the maximum log-likelihood objective function.
(b) Derive the gradient of the maximum log-likelihood objective.
(c) Derive the gradient descent update rule for learning Θ.
(d) Programming. In this programming assignment, you are going to apply the multi-class logistic regression
to handwritten digits classification problem. We provided a subset of USPS datasets which contains
three classes: ‘1’, ‘2’, ‘3’. Please implement a multi-class logistic regression with gradient update rule on this
dataset.

The dataset ‘usps 3cls.mat’ contains four matrices.
1. ‘xtrain’, 2700 × 256 matrix, is the training data with 2700 instances and each of them is with 256
features.
2. ‘xtest’, 600 × 256 matrix, is the training data with 600 instances.
3. ‘ytrain’, 2700 × 1 matrix, is the labels of training data. The labels can take three values 1, 2, 3.
4. ‘ytest’, 600 × 1 matrix, is the labels of testing data. The labels can take three values 1, 2, 3

In myRegressions.py, please implement the function
‘def multiclass logistic regression(xtrain, ytrain, xtest, ytest, stepsize, max iter) ’

Input
‘xtrain’, ‘ytrain’, ‘xtest’ and ‘ytest’ are the training and testing dataset.
‘stepsize’ is the learning rate.
‘max iter´ıs the maximum iteration numbers for updating.

Output
‘W’, D × K matrix where D is the number of features, is the learned parameters of the multi-class
logistic regression.
‘trainerr’, max iter × 1 vector, contains the training error after each update.
‘testerr’, max iter × 1 vector, contains the testing error after each update.

2 Programming: Recommendation System
Personalized recommendation systems are used in a wide variety of applications such as electronic commerce,
social networks, web search, and more. Machine learning techniques play a key role to extract individual
preference over items. In this assignment, we explore this popular business application of machine learning,
by implementing a simple matrix-factorization-based recommender using gradient descent.
Suppose you are an employee in Netflix. You are given a set of ratings (from one star to five stars)
from users on many movies they have seen. Using this information, your job is implementing a personalized
rating predictor for a given user on unseen movies. That is, a rating predictor can be seen as a function
f : U × I → R, where U and I are the set of users and items, respectively. Typically the range of this
function is restricted to between 1 and 5 (stars), which is the the allowed range of the input.
Now, let’s think about the data representation. Suppose we have m users and n items, and a rating
given by a user on a movie. We can represent this information as a form of matrix, namely rating matrix M.
Suppose rows of M represent users, while columns do movies. Then, the size of matrix will be m × n. Each
cell of the matrix may contain a rating on a movie by a user. In M15,47, for example, it may contain a rating
on the item 47 by user 15. If he gave 4 stars, M15,47 = 4. However, as it is almost impossible for everyone to
watch large portion of movies in the market, this rating matrix should be very sparse in nature. Typically,
only 1% of the cells in the rating matrix are observed in average. All other 99% are missing values, which
means the corresponding user did not see (or just did not provide the rating for) the corresponding movie.
Our goal with the rating predictor is estimating those missing values, reflecting the user’s preference learned
from available ratings.

(c) Implement def my recommender(rate matrix, rank) by filling the gradient descent part.
You are given a skeleton function def my recommender(rate matrix, rank) in myRegressions.py. Using
the training data rate matrix, you will implement your own recommendation system of rank low rank.
In the gradient descent part, repeat your update formula in (b), observing the average reconstruction error
between your estimation and ground truth in training set. You need to set a stopping criteria, based on this
reconstruction error as well as the maximum number of iterations. You should play with several different
values for µ and λ to make sure that your final prediction is accurate.
Formatting information is here:

Input
• rateMatrix: training data set. Each row represents a user, while each column an item. Observed
values are one of {1, 2, 3, 4, 5}, and missing values are 0.
• lowRank: the number of factors (dimension) of your model. With higher values, you would expect
more accurate prediction.

Output
• U: the user profile matrix of dimension user count × low rank.
• V: the item profile matrix of dimension item count × low rank.
