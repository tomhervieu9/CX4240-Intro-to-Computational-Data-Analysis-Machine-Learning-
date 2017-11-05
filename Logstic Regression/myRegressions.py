import numpy as np
import matplotlib.pyplot as plt

def multiclass_logistic_regression(xtrain, ytrain, xtest, ytest, stepsize, total_iter):
    """

    :param xtrain:
    :param ytrain:
    :param xtest:
    :param ytest:
    :param stepsize: learning rate
    :param total_iter: maximum number of iterations for updating
    :return:
    """
#Initialize_________________________________________________________
    W = np.zeros((xtrain.shape[1], np.max(ytrain)))
    trainerr = []
    testerr = []
    
    flag=np.zeros((np.max(ytrain), xtrain.shape[0]))
    grad = np.zeros((np.max(ytrain), xtrain.shape[0]))

    xtrainZ= (xtrain-np.mean(xtrain,axis=0))/np.std(xtrain,axis=0)
    xtestZ= (xtest-np.mean(xtrain,axis=0))/np.std(xtrain,axis=0)

#Adjust to Indexed Columns__________________________________________
    ytest = ytest - 1
    ytrain = ytrain - 1

#1 where a value exists in Sparse Matrix____________________________
    for x in range (xtrain.shape[0]):
        flag[ytrain[x], x] = 1

    x = 0
    while (x < total_iter):
        trainErrVector = np.zeros((xtrain.shape[0],1)) #Initialize error vectors
        testErrVector = np.zeros((xtest.shape[0],1))

        #Apply regression formula
        probExpected = np.exp((W.T).dot(xtrainZ.T))
        probActual = np.exp((W.T).dot(xtestZ.T))

        meanProbExpected = np.nan_to_num(probExpected / np.sum(probExpected, axis = 0))
        meanProbActual = np.nan_to_num(probActual / np.sum(probActual, axis = 0))

        grad = (meanProbExpected-flag).dot(xtrainZ)
        diff = stepsize*grad.T
        W = W - diff

        #take the max probability value
        trainLabel = np.argmax(meanProbExpected, axis = 0).reshape(xtrain.shape[0],1)
        testLabel = np.argmax(meanProbActual, axis = 0).reshape(xtest.shape[0],1)

        #Compare training and testing sets to form error vectors
        trainErrVector[np.equal(ytrain, trainLabel)] = 1
        testErrVector[np.equal(ytest, testLabel)] = 1

        #calculate error and assing to the 2 arrays we are returning
        trainErr = 1 - (np.sum(trainErrVector) / xtrain.shape[0])
        testErr = 1 - (np.sum(testErrVector) / xtest.shape[0])

        trainerr.append(trainErr)
        testerr.append(testErr)
        x = x + 1
#Plot errors-________________________________________________________
    plt.plot(trainerr)
    plt.plot(testerr)
    plt.ylabel('Error')
    plt.xlabel('Iterations')
    plt.show()

    return W, trainerr, testerr


def my_recommender(rate_matrix, low_rank):
    """

    :param rate_matrix:
    :param low_rank:
    :return:
    """

    # Parameters
    maxIter = 0 # CHOOSE YOUR OWN
    learningRate = 0 # CHOOSE YOUR OWN
    regularizer = 0 # CHOOSE YOUR OWN

    U = None # PERFORM INITIALIZATION
    V = None # PERFORM INITIALIZATION

    # PERFORM grad DESCENT
    # IMPLEMENT YOUR CODE HERE

    return U, V
