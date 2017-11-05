"""
    CX4240 Spring 2017
    Homework 4
    Suggestions are welcome!
"""

import scipy.io as sio
import time
import numpy as np
from sklearn.metrics import mean_squared_error
from math import sqrt

from myRegressions import multiclass_logistic_regression, my_recommender


if __name__ == '__main__':

    ##### PART 1 :: Multiclass Logistic Regression
    usps_data = sio.loadmat('usps_3cls.mat')

    xtrain = usps_data['xtrain']
    ytrain = usps_data['ytrain']
    xtest = usps_data['xtest']
    ytest = usps_data['ytest']

    W, trainerr, testerr = multiclass_logistic_regression(xtrain=xtrain,
                                                          ytrain=ytrain,
                                                          xtest=xtest,
                                                          ytest=ytest,
                                                          stepsize=0.001,
                                                          total_iter=500)

    print("PART 1. MULTICLASS LOGISTIC REGRESSION")
    print("TRAIN ERROR = {}, TEST ERRORS = {}".format(trainerr, testerr))

    ##### PART 2 :: Recommender System with Matrix Factorization
    print("PART 2. MATRIX FACTORIZATION")
    movie_data = sio.loadmat('movie_data.mat')
    rateMatrix = movie_data['train']
    testMatrix = movie_data['test']

    low_ranks = [1, 3, 5]
    for lr in low_ranks:
        start_time = time.time()
        U, V = my_recommender(rateMatrix, int(lr))
        end_time = time.time()
        rmse_train = sqrt(mean_squared_error(rateMatrix, np.dot(U, V.T)))
        rmse_test = sqrt(mean_squared_error(testMatrix, np.dot(U, V.T)))

        print("FOR RANK = {}, RMSE_TRAIN = {}, RMSE_TEST = {}, RUNTIME = {}".format(lr, rmse_train, rmse_test, end_time - start_time))
