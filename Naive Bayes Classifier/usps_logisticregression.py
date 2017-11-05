############################################################################
# The codes are based on Python2.7. 
# Please install numpy, scipy, matplotlib before using.
# Thank you for your suggestions!
#
# @version 1.0
# @author CSE6740/CS7641 TA
############################################################################
## 
# load handwritten digit dataset
import scipy.io as sio 
import numpy as np
import matplotlib.pyplot as plt
import show_image
from scipy import spatial
from scipy import stats
from scipy.sparse import csc_matrix
import logistic_regression
import warnings
warnings.filterwarnings("ignore")

load1 = sio.loadmat('usps_all.mat')
data = load1['data'].astype(float)
# placing zero first

train_size = np.round(0.8 * data.shape[1]).astype(int)
test_size = data.shape[1] - train_size

#np.random.seed(1)

rno = 1
nb_accuracy = np.zeros((1, rno))
knn_accuracy = np.zeros((1, rno))
accuracy = np.zeros((1, rno))
for r in range(0, rno):
    
    indices = np.random.permutation(data.shape[1])
    
    indices_train = indices[0:train_size]
    indices_test = indices[train_size:data.shape[1]]
        
    num = 3
    trainx = np.concatenate((data[:, indices_train, 9], data[:, indices_train, 0] \
                            , data[:, indices_train, 1] \
    #                       , data[:, indices_train, 2] \
    #                       , data[:, indices_train, 3], data[:, indices_train, 4] \
    #                       , data[:, indices_train, 5], data[:, indices_train, 6] \
    #                       , data[:, indices_train, 7], data[:, indices_train, 8]\
                            ), axis = 1)
    trainy = np.concatenate((np.zeros((1, train_size)), np.ones((1, train_size)) \
                            , 2 * np.ones((1, train_size)) \
    #                       , 3 * np.ones((1, train_size)) \
    #                       , 4 * np.ones((1, train_size)), 5 * np.ones((1, train_size)) \
    #                       , 6 * np.ones((1, train_size)), 7 * np.ones((1, train_size)) \
    #                       , 8 * np.ones((1, train_size)), 9 * np.ones((1, train_size)) \
                            ), axis = 1)
    trainno = np.shape(trainy)[1]
    dimno = np.shape(trainx)[0]
    
    testx = np.concatenate((data[:, indices_test, 9], data[:, indices_test, 0] \
                            , data[:, indices_test, 1] \
    #                       , data[:, indices_test, 2] \
    #                       , data[:, indices_test, 3], data[:, indices_test, 4] \
    #                       , data[:, indices_test, 5], data[:, indices_test, 6] \
    #                       , data[:, indices_test, 7], data[:, indices_test, 8]\
                            ), axis = 1)
    testy = np.concatenate((np.zeros((1, test_size)), np.ones((1, test_size)) \
                            , 2 * np.ones((1, test_size)) \
    #                       , 3 * np.ones((1, test_size)) \
    #                       , 4 * np.ones((1, test_size)), 5 * np.ones((1, test_size)) \
    #                       , 6 * np.ones((1, test_size)), 7 * np.ones((1, test_size)) \
    #                       , 8 * np.ones((1, test_size)), 9 * np.ones((1, test_size)) \
                            ), axis = 1)
    testno = np.shape(testy)[1]
    
    plt.figure()
    plt.ion()
    show_image.show_image_function(testx.T, 16, 16)
    plt.axis('off')
    plt.show()
    plt.ioff()
            
    raw_input('press a key to continue ....\n')
    plt.close("all")
    
    print 'please wait for Naive Bayes!'
    
    ##
    # Naive Bayes
        
    # estimate class prior distribution; 
    py = np.zeros((num, 1))
    for i in range(0,num):
        py[i, 0] = np.sum(trainy == i) * 1.0 / trainno 
    
    # estimate the class conditional distribution; 
    mu_x_y = np.zeros((dimno, num))
    sigma_x_y = np.zeros((dimno, num))
    
    for j in range(0, num):
        for i in range(0, dimno):
            # taking data points from appropriate class for each dimension and finding mean and variance
            mu_x_y[i, j] = np.mean(trainx[i, j * train_size + np.arange(train_size)]) 
            sigma_x_y[i, j] = np.std(trainx[i, j * train_size + np.arange(train_size)], ddof = 1)
                    
    pytest = np.zeros((testno, num))
    predy = np.zeros((testno, 1))
    img_class = np.zeros((num, dimno, testno))
    count_acc = 0
    for i in range(0, testno):
        
        # for each class
        # for each and every dimension, we sum the predicted probability based
        # on mean and variance calculated from training data
        # and prior from training data
        for k in range(0, num):
            pytest[i, k] = np.log10(py[k, 0])
            for j in range(0, dimno):
                with np.errstate(divide='ignore'):
                    pytest[i, k] = pytest[i, k] + np.log10(stats.norm.pdf(testx[j,i], mu_x_y[j,k], sigma_x_y[j,k] + np.power(0.1, 3)))
            
        # select maximum probability among classes
        index = np.nanargmax(pytest[i,:])
        predy[i, 0] = index
        
        if(predy[i, 0] == testy[0, i]):
            count_acc = count_acc + 1;
        
        img_class[index, :, i] = testx[:, i]
            
    nb_accuracy[0, r] = count_acc * 1.0 / testno
    print 'nb: '
    print nb_accuracy[0, r]    
    
    if r == 0:
        # plot all USPS digits predicted by class
        for i in range(0, num):
            arr = img_class[i, :, :]
            
            plt.figure()
            plt.ion()
            show_image.show_image_function(arr.T, 16, 16)
            plt.axis('off')
            plt.title('naive bayes digit: ' + str(i))
            plt.show()
            plt.ioff()
                          
    raw_input('press a key to continue ....\n')
    plt.close("all")
    
    print 'please wait for K-nearest neighbor!'
               
    ## K-nearest neighbor
    
    # For each test point find the nearest 10 training points and classify
    # accordingly
    predy_knn = np.zeros((testno, 1))
    img_class_knn = np.zeros((num, dimno, testno))
    count_acc_knn = 0
    KDtree = spatial.cKDTree(trainx.T) 
    for i in range(0, testno):
        
        # returns a n cross 10 matrix of indices, if looking for 10 nearest neighbors
        n = KDtree.query(testx[:, i], k = 10)[1]
        
        count = np.zeros((1, 10))
        
        # incrementing count, since trainy can contain zero, incrementing by 1 
        for j in range(0, n.shape[0]):
            count[0, trainy[0, n[j]].astype(int) ] = count[0, trainy[0, n[j]].astype(int) ] + 1
        
        index = np.argmax(count[0, :])
        predy_knn[i, 0] = index
        
        if(predy_knn[i, 0] == testy[0, i]):
            count_acc_knn = count_acc_knn + 1;
        
        img_class_knn[index, :, i] = testx[:, i]  
            
    knn_accuracy[0, r] = count_acc_knn * 1.0 / testno
    print 'knn: '
    print knn_accuracy[0, r]
    
    if r == 0:
        # plot all images predicted by class
        for i in range(0, num):
            arr = img_class_knn[i, :, :]
            plt.figure()
            plt.ion()
            show_image.show_image_function(arr.T, 16, 16)
            plt.axis('off') 
            plt.title('knn digit: ' + str(i))
            plt.show()
            plt.ioff() 
        
    raw_input('press a key to continue ....\n')
    plt.close("all")
    
    print 'please wait for Logistic Regression!'
    
    ## logistic regression; 
    trainX = np.concatenate((trainx, np.ones((1, trainno))), axis = 0)
    testX = np.concatenate((testx, np.ones((1, testno))), axis = 0)
    dimno = trainX.shape[0]
        
    trainY = csc_matrix( (np.ones(trainno), (np.arange(trainno), trainy.squeeze())), shape=(trainno, num))
    reg_param = np.power(0.1, 5)
    iters = 200
    show_error = 1
    theta, train_error = logistic_regression.log_regress_train(trainX.T, trainY, reg_param, iters, show_error)
        
    pytest = np.zeros((testno, num))
    predy = np.zeros((testno, 1))
    img_class = np.zeros((num, dimno - 1, testno))
    count_acc = 0
    for i in range(0, testno):
        # for each class
        # for each and every dimension, we sum the predicted probability based
        # on mean and variance calculated from training data
        # and prior from training data
        for k in range(0, num):
            tmp1 = theta[:, k].reshape(theta[:, k].shape[0], 1)
            tmp2 = testX[:, i].reshape(testX[:, i].shape[0], 1)
            pytest[:, k] = np.exp(tmp1.T.dot(tmp2))
            
        # select maximum probability among classes
        index = np.argmax(pytest[i, :])
        predy[i, 0] = index
        
        if predy[i, 0] == testy[0, i]:
            count_acc = count_acc + 1
        
        img_class[index, :, i] = testX[0 : testX.shape[0] - 1, i]
    
    accuracy[0, r] = count_acc * 1.0 / testno
    
    if r == 0:
        # plot all USPS digits predicted by class
        for i in range(0, num):
            arr = img_class[i, :, :]
            plt.figure()
            plt.ion()
            show_image.show_image_function(arr.T, 16, 16)
            plt.axis('off') 
            plt.title('logistic regression digit: ' + str(i))
            plt.show()
            plt.ioff() 
    
    raw_input('press a key to continue ....\n')
    plt.close("all")
            
nb_accuracy_final = np.mean(nb_accuracy.squeeze())
print 'nb_final: '
print nb_accuracy_final   
knn_accuracy_final = np.mean(knn_accuracy.squeeze())   
print 'knn_final: '
print knn_accuracy_final
lg_accuracy_final = np.mean(accuracy.squeeze())   
print 'lg_final: '
print lg_accuracy_final
#plt.show()
