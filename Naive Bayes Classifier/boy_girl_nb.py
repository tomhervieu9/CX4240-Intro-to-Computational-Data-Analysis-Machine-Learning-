############################################################################
# The codes are based on Python2.7. 
# Please install numpy, scipy, matplotlib before using.
# Thank you for your suggestions!
#
# @version 1.0
# @author CSE6740/CS7641 TA
############################################################################
import scipy.io as sio 
import numpy as np
import matplotlib.pyplot as plt
import show_image
from scipy import stats
import warnings
warnings.filterwarnings("ignore")

# load dataset for boys and girls
# dividing into 5 equally split datasets for cross validation
# choosing five fold split as 10 results in highly uneven splits
load1 = sio.loadmat('boys.mat')
boys_data = load1['data'].astype(float)
load2 = sio.loadmat('girls.mat')
girls_data = load2['data'].astype(float) 

train_size_boys = boys_data.shape[1]
train_size_girls = girls_data.shape[1]

test_size_boys = train_size_boys
test_size_girls = train_size_girls

np.random.seed(1)

# iterating through random splits of data
pno = 1
nb_accuracy = np.zeros((1, pno))
for p in range(0,pno):
    
    
    boys_indices = np.random.permutation(boys_data.shape[1])
    girls_indices = np.random.permutation(girls_data.shape[1])
        
    boys_train = boys_indices[0:train_size_boys]
    girls_train = girls_indices[0:train_size_girls]
    
    boys_test = boys_train
    girls_test = girls_train
    
    trainx = np.concatenate((boys_data[:,boys_train],girls_data[:, girls_train]) , axis = 1)
    testx = np.concatenate((boys_data[:,boys_test],girls_data[:, girls_test]), axis = 1)
    trainy = np.concatenate((np.zeros((1, train_size_boys)), np.ones((1, train_size_girls))), axis = 1)
    testy = np.concatenate((np.zeros((1, test_size_boys)), np.ones((1, test_size_girls))), axis = 1)
    
    #plt.figure(1)
    #show_image.show_image_function(trainx.T, 65, 65)
    trainno = np.shape(trainy)[1]
    dimno = np.shape(trainx)[0]
    testno = np.shape(testy)[1]
    
    plt.figure(1)
    show_image.show_image_function(testx.T, 65, 65)
    plt.axis('off')
              
    print 'please wait for Naive Bayes!'       
    ## 
    # Naive Bayes
    
    # estimate class prior distribution; 
    py = np.zeros((2, 1))
    for i in range(0,2):
        py[i, 0] = np.sum(trainy == i) * 1.0 / trainno 
        
    # estimate the class conditional distribution; 
    mu_x_y = np.zeros((dimno, 2))
    sigma_x_y = np.zeros((dimno, 2))
    for i in range(0, dimno):
        # taking data points from appropriate class for each dimension and finding mean and variance
        mu_x_y[i, 0] = np.mean(trainx[i, 0:train_size_boys])
        mu_x_y[i, 1] = np.mean(trainx[i, train_size_boys: train_size_boys + train_size_girls])
        sigma_x_y[i, 0] = np.std(trainx[i, 0:train_size_boys], ddof=1)
        sigma_x_y[i, 1] = np.std(trainx[i, train_size_boys:train_size_boys + train_size_girls], ddof = 1)
    
    pytest = np.zeros((testno, 2))
    predy = np.zeros((testno, 1))
    img_class = np.zeros((2, dimno, testno))
    count_acc = 0
    for i in range(0, testno):
    
        # for each class
        # for each and every dimension, we sum the predicted probability based
        # on mean and variance calculated from training data
        # and prior from training data
        for k in range(0, 2):
            pytest[i, k] = np.log10(py[k, 0])
            for j in range(0, dimno):
                pytest[i, k] = pytest[i, k] + np.log10(stats.norm.pdf(testx[j,i], mu_x_y[j,k], sigma_x_y[j,k] + np.power(0.1, 3)))
    
        # select maximum probability among classes
        index = np.argmax(pytest[i,:])
    
        predy[i, 0] = index
       
        if(predy[i, 0] == testy[0, i]):
            count_acc = count_acc + 1;
        
        img_class[index, :, i] = testx[:, i]
    
    nb_accuracy[0,p] = count_acc * 1.0 / testno
    print nb_accuracy[0,p]
    
    if p == 0:
        # plot all images predicted by class
        arr = img_class[0, :, :]
        plt.figure(2)
        show_image.show_image_function(arr.T, 65, 65)
        plt.title('boys')
        plt.axis('off')
        
        arr = img_class[1, :, :]
        plt.figure(3)
        show_image.show_image_function(arr.T, 65, 65)
        plt.title('girls')
        plt.axis('off')
        plt.show()