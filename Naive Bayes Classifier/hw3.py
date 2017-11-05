"""
    Main routine for Homework 3, CX4240 - Spring 2017
    The code is based on Python 2.7
    Please install numpy, scipy before using.
    Thank you for your suggestions!

    @version 1.0
    @author: Jenna Kwon
"""

import numpy as np
import scipy.io as sio
import random
from classifiers import modelFull, modelSpherical, modelDiagional


def splitData(filename, testing_set_percentage):
    """
        Randomly split data into training set, testing set based on given ratio

    :param filename: .mat file containing dataset
    :return: train, testing dataset
    """
    matFile = sio.loadmat(filename)
    data = matFile['mydata']

    np.savetxt('test.out', data, fmt='%d', delimiter=',')  # X is an array
    print('BEFORE SHUFFLE', data)

    np.random.shuffle(data)

    np.savetxt('test_after.out', data, fmt='%d', delimiter=',')  # X is an array

    print('AFTER SHUFFLE', data)

    r, c = np.array(data).shape

    testing_set_size = int(r * testing_set_percentage)
    training_set_size = r - testing_set_size

    train_data = data[:training_set_size]
    test_data = data[training_set_size:]

    return train_data, test_data


if __name__ == '__main__':
    # repeat the experiments 100 times
    N = 1

    # 10-90 train-test split, 20-80 train-test split, ...., 90-10 train-test splits.
    # splits = [0.1, 0.2, 0.5, 0.8, 0.9]
    splits = [0.1]

    for p in splits:

        # initialize error matrices
        err_full = np.zeros((N, 2))
        err_diagonal = np.zeros((N, 2))
        err_spherical = np.zeros((N, 2))

        for i in range(0, N):
            train, test = splitData('usps_2cls.mat', p)

            err_train_1, err_test_1 = modelFull(train, test)
            err_train_2, err_test_2 = modelDiagional(train, test)
            err_train_3, err_test_3 = modelSpherical(train, test)

            err_full[i, :] = [err_train_1, err_test_1]
            err_diagonal[i, :] = [err_train_2, err_test_2]
            err_spherical[i, :] = [err_train_3, err_test_3]

        # # 2-dimensional vectors for average errors
        mean_error_full = np.mean(err_full, axis=0)
        mean_error_diagonal = np.mean(err_diagonal, axis=0)
        mean_error_spherical = np.mean(err_spherical, axis=0)

        print("\n============= FOR p = %s ============ " % p)
        print('\n error_full for train, test : %s %s' % (mean_error_full[0], mean_error_full[1]))
        print('\n err_diagonal for train, test : %s %s' % (mean_error_diagonal[0], mean_error_diagonal[1]))
        print('\n err_spherical for train, test : %s %s' % (mean_error_spherical[0], mean_error_spherical[1]))
