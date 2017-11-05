"""
	Implement your functions here for Bayes classification.clear

	Each function takes in training and testing sets as input,
	and returns accuracy on classification for training and testing sets, respectively.
"""
import numpy as np

def modelDiagional(train, test):
	# Initializations________________________________________________
    err_train = 0
    err_test = 0
    vec0 = []
    vec1 = []
    train_set = []
    test_set = []
    classes = []
    train_classes = []
    test_classes = []
    train_results = []
    test_results = []
    numTrain = 0
    numTest = 0


    # Forming testing and training sets______________________________
    for x in train:
        train_set.append(x[:-1]/255)
        classes.append(x[-1])
        train_classes.append(x[-1])
    for x in test:
        test_set.append(x[:-1]/255)
        classes.append(x[-1])
        test_classes.append(x[-1])
    for x in train:
        if x[-1] == 0:
            vec0.append(x[:-1]/255)
        if x[-1] == 1:
            vec1.append(x[:-1]/255)

    #reformatting to arrays__________________________________________
    train_set = np.array(train_set) #reformatted to perform np operations


    #Building our covariance matrix__________________________________
    newVec0 = float(len(vec0))/float(len(vec0) + len(vec1))
    newVec1 = float(len(vec1))/float(len(vec0) + len(vec1))

    vec0 = np.array(vec0)
    vec1 = np.array(vec1)

    cov_x0 = np.cov(vec0.T)*(np.eye(256)) + (np.eye(256)/100)
    cov_x1 = np.cov(vec1.T)*(np.eye(256)) + (np.eye(256)/100)

    #Likelihood function
    cov1Det = - float(np.linalg.slogdet(cov_x1)[1])
    cov0Det = - float(np.linalg.slogdet(cov_x0)[1])

    #Forming resulting vectors for training set______________________
    for x in train_set:
        diff0 = x - vec0.mean(axis=0)
        diff1 = x - vec1.mean(axis=0)
        inv0 = np.linalg.inv(cov_x0)
        inv1 = np.linalg.inv(cov_x1)
        vec0Prob = cov0Det - ((diff0.T).dot(inv0)).dot(diff0)
        vec1Prob = cov1Det - ((diff1.T).dot(inv1)).dot(diff1)
        if (vec0Prob + np.log(newVec0))/(vec1Prob + np.log(newVec1)) >= 1:
            train_results.append(0)
        else:
            train_results.append(1)

    train_results = np.array(train_results)
    for x in range(train_results.shape[0]):
        if int(train_results[x]) == int(train_classes[x]):
            numTrain += 1

    #Forming resulting vectors for testing set______________________
    for x in test_set:
        diff0 = x - vec0.mean(axis=0)
        diff1 = x - vec1.mean(axis=0)
        inv0 = np.linalg.inv(cov_x0)
        inv1 = np.linalg.inv(cov_x1)
        vec0Prob = cov0Det - ((diff0.T).dot(inv0)).dot(diff0)
        vec1Prob = cov1Det - ((diff1.T).dot(inv1)).dot(diff1)
        if (vec0Prob + np.log(newVec0))/(vec1Prob + np.log(newVec1)) >= 1:
            test_results.append(0)
        else:
            test_results.append(1)

    test_results = np.array(test_results)
    for x in range(test_results.shape[0]):
        if int(test_results[x]) == int(test_classes[x]):
            numTest += 1


    # calculating error for training and testing sets
    err_test =  1 - (float(numTest)/ test_results.shape[0])
    err_train =  1 - (float(numTrain)/ train_results.shape[0])

    return err_train, err_test


def modelFull(train, test):
    # Initializations________________________________________________
    err_train = 0
    err_test = 0
    vec0 = []
    vec1 = []
    train_set = []
    test_set = []
    classes = []
    train_classes = []
    test_classes = []
    train_results = []
    test_results = []
    numTrain = 0
    numTest = 0


    # Forming testing and training sets______________________________
    for x in train:
        train_set.append(x[:-1]/255)
        classes.append(x[-1])
        train_classes.append(x[-1])
    for x in test:
        test_set.append(x[:-1]/255)
        classes.append(x[-1])
        test_classes.append(x[-1])
    for x in train:
        if x[-1] == 0:
            vec0.append(x[:-1]/255)
        if x[-1] == 1:
            vec1.append(x[:-1]/255)

    #reformatting to arrays__________________________________________
    train_set = np.array(train_set) #reformatted to perform np operations


    #Building our covariance matrix__________________________________
    newVec0 = float(len(vec0))/float(len(vec0) + len(vec1))
    newVec1 = float(len(vec1))/float(len(vec0) + len(vec1))

    vec0 = np.array(vec0)
    vec1 = np.array(vec1)

    cov_x0 = np.cov(vec0.T) + (np.eye(256)/100)
    cov_x1 = np.cov(vec1.T) + (np.eye(256)/100)

    cov1Det = - float(np.linalg.slogdet(cov_x1)[1])
    cov0Det = - float(np.linalg.slogdet(cov_x0)[1])

    #Forming resulting vectors for training set______________________
    for x in train_set:
        diff0 = x - vec0.mean(axis=0)
        diff1 = x - vec1.mean(axis=0)
        inv0 = np.linalg.inv(cov_x0)
        inv1 = np.linalg.inv(cov_x1)
        vec0Prob = cov0Det - ((diff0.T).dot(inv0)).dot(diff0)
        vec1Prob = cov1Det - ((diff1.T).dot(inv1)).dot(diff1)
        if (vec0Prob + np.log(newVec0))/(vec1Prob + np.log(newVec1)) >= 1:
            train_results.append(0)
        else:
            train_results.append(1)

    train_results = np.array(train_results)
    for x in range(train_results.shape[0]):
        if int(train_results[x]) == int(train_classes[x]):
            numTrain += 1

    #Forming resulting vectors for testing set______________________
    for x in test_set:
        diff0 = x - vec0.mean(axis=0)
        diff1 = x - vec1.mean(axis=0)
        inv0 = np.linalg.inv(cov_x0)
        inv1 = np.linalg.inv(cov_x1)
        vec0Prob = cov0Det - ((diff0.T).dot(inv0)).dot(diff0)
        vec1Prob = cov1Det - ((diff1.T).dot(inv1)).dot(diff1)
        if (vec0Prob + np.log(newVec0))/(vec1Prob + np.log(newVec1)) >= 1:
            test_results.append(0)
        else:
            test_results.append(1)

    test_results = np.array(test_results)
    for x in range(test_results.shape[0]):
        if int(test_results[x]) == int(test_classes[x]):
            numTest += 1


    # calculating error for training and testing sets
    err_test =  1 - (float(numTest)/ test_results.shape[0])
    err_train =  1 - (float(numTrain)/ train_results.shape[0])

    return err_train, err_test


def modelSpherical(train, test):
    # Initializations________________________________________________
    err_train = 0
    err_test = 0
    vec0 = []
    vec1 = []
    train_set = []
    test_set = []
    classes = []
    train_classes = []
    test_classes = []
    train_results = []
    test_results = []
    numTrain = 0
    numTest = 0


    # Forming testing and training sets______________________________
    for x in train:
        train_set.append(x[:-1]/255)
        classes.append(x[-1])
        train_classes.append(x[-1])
    for x in test:
        test_set.append(x[:-1]/255)
        classes.append(x[-1])
        test_classes.append(x[-1])
    for x in train:
        if x[-1] == 0:
            vec0.append(x[:-1]/255)
        if x[-1] == 1:
            vec1.append(x[:-1]/255)

    #reformatting to arrays__________________________________________
    train_set = np.array(train_set) #reformatted to perform np operations


    #Building our covariance matrix__________________________________
    newVec0 = float(len(vec0))/float(len(vec0) + len(vec1))
    newVec1 = float(len(vec1))/float(len(vec0) + len(vec1))

    vec0 = np.array(vec0)
    vec1 = np.array(vec1)

    cov_x0 = np.mean(np.diag(np.cov(vec0.T)))*(np.eye(256))
    cov_x1 = np.mean(np.diag(np.cov(vec1.T)))*(np.eye(256))

    cov1Det = - float(np.linalg.slogdet(cov_x1)[1])
    cov0Det = - float(np.linalg.slogdet(cov_x0)[1])

    #Forming resulting vectors for training set______________________
    for x in train_set:
        diff0 = x - vec0.mean(axis=0)
        diff1 = x - vec1.mean(axis=0)
        inv0 = np.linalg.inv(cov_x0)
        inv1 = np.linalg.inv(cov_x1)
        vec0Prob = cov0Det - ((diff0.T).dot(inv0)).dot(diff0)
        vec1Prob = cov1Det - ((diff1.T).dot(inv1)).dot(diff1)
        if (vec0Prob + np.log(newVec0))/(vec1Prob + np.log(newVec1)) >= 1:
            train_results.append(0)
        else:
            train_results.append(1)

    train_results = np.array(train_results)
    for x in range(train_results.shape[0]):
        if int(train_results[x]) == int(train_classes[x]):
            numTrain += 1

    #Forming resulting vectors for testing set______________________
    for x in test_set:
        diff0 = x - vec0.mean(axis=0)
        diff1 = x - vec1.mean(axis=0)
        inv0 = np.linalg.inv(cov_x0)
        inv1 = np.linalg.inv(cov_x1)
        vec0Prob = cov0Det - ((diff0.T).dot(inv0)).dot(diff0)
        vec1Prob = cov1Det - ((diff1.T).dot(inv1)).dot(diff1)
        if (vec0Prob + np.log(newVec0))/(vec1Prob + np.log(newVec1)) >= 1:
            test_results.append(0)
        else:
            test_results.append(1)

    test_results = np.array(test_results)
    for x in range(test_results.shape[0]):
        if int(test_results[x]) == int(test_classes[x]):
            numTest += 1


    # calculating error for training and testing sets
    err_test =  1 - (float(numTest)/ test_results.shape[0])
    err_train =  1 - (float(numTrain)/ train_results.shape[0])

    return err_train, err_test
