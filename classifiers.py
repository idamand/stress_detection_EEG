from matplotlib import pyplot as plt
import numpy as np
#from sklearn.metrics import plot_confusion_matrix

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from metrics import compute_metrics
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import RobustScaler, StandardScaler

from EEGNet.EEGModels import EEGNet
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
#from keras.utils import np_utils
#from keras.callbacks import ModelCheckpoint
from pyriemann.utils.viz import plot_confusion_matrix

import utils.variables as var
import mne


def svm(train_data, test_data, train_labels, test_labels):
    '''
    Parameters
    ----------
    train_data : dict
        Path to the file to be read.
    test_data : dict
        Test data
    train_labels : dict
        Labels of the train data
    test_labels : dict
        Labels of the test data

    Returns
    -------
    metrics : confusion matrix
        The confusion matrix with the results 
    '''

    param_grid = {
        'C': [0.01, 0.1, 1, 10, 100, 1000, 10000],
        'kernel': ['poly', 'sigmoid', 'linear', 'rbf']
    }

    scaler = StandardScaler()
    train_data = scaler.fit_transform(train_data)
    test_data = scaler.transform(test_data)

    #weights = {0:67, 1:33}
    #scaler = RobustScaler()
    #train_data = scaler.fit_transform(train_data)
    #test_data = scaler.transform(test_data)

    svm_clf = GridSearchCV(SVC(), param_grid=param_grid, refit=True, n_jobs=-1, cv=10)
    svm_clf.fit(train_data, train_labels)

    y_pred = svm_clf.predict(test_data)
    y_true = test_labels

    cv_results = svm_clf.cv_results_
    accuracy = cv_results['mean_test_score']
    #print('--------------------- RESULTS FROM GRIDSEARCH --------------------- \n', cv_results)
    print('--------------------- BEST PARAMETERS FROM GRIDSEARCH --------------------- \n', svm_clf.best_params_)
    print(' OVERALL ACCURACY:', np.round(np.sum(accuracy)/len(accuracy)*100,2))

    plt.figure(1)
    plt.plot(accuracy)
    plt.xlabel('Fold')
    plt.ylabel('Mean accuracy of test score')
    plt.show()

    metrics = compute_metrics(y_true, y_pred)

    return metrics


def rf(train_data, test_data, train_labels, test_labels):
    '''
    Input: data of shape (n_samples, n_features), and labels of shape (n_samples). Performs random forest classification
    '''

    param_grid = {
        'n_estimators' : [100, 200, 300, 400, 500],
        'criterion' : ['gini', 'entropy'],
        'max_features' : ['auto', 'sqrt', 'log2'],
        'max_depth' : [70, 80, 90, 100, 'None']
    }
    
    #weights = {0:67, 1:33}
    
    rf_clf = GridSearchCV(RandomForestClassifier(), param_grid=param_grid, refit=True, n_jobs=-1, cv=10)
    rf_clf.fit(train_data, train_labels)

    y_pred = rf_clf.predict(test_data)
    y_true = test_labels

    cv_results = rf_clf.cv_results_
    accuracy = cv_results['mean_test_score']
    #print('--------------------- RESULTS FROM GRIDSEARCH --------------------- \n', cv_results)
    print('--------------------- BEST PARAMETERS FROM GRIDSEARCH --------------------- \n', rf_clf.best_params_)
    print(' OVERALL ACCURACY:', np.round(np.sum(accuracy)/len(accuracy)*100,2))

    plt.figure(1)
    plt.plot(accuracy)
    plt.xlabel('Fold')
    plt.ylabel('Mean accuracy of test score')
    plt.show()

    metrics = compute_metrics(y_true, y_pred)


    return metrics

def knn(train_data, test_data, train_labels, test_labels):
    '''
    Explanation

    Parameters
    ----------
    train_data : ndarray
        An array containing the training data, shape(n_recordings, n_channels*n_features)
    test_data : ndarray
        An array containing the test data, shape(n_recordings, n_channels*n_features)
    train_labels : ndarray
        An array containing the labels of the training set, shape(n_recordings, )

    Returns
    -------
    metrics : something

    '''
    param_grid = {
        'n_neighbors': [1, 3, 5, 7, 9],
        'leaf_size': [5, 10, 20, 30, 40, 50],
        'p': [1, 2]
    }

    knn_clf = GridSearchCV(KNeighborsClassifier(), param_grid, refit=True, n_jobs=-1, cv = 10)
    knn_clf.fit(train_data, train_labels)

    y_pred = knn_clf.predict(test_data)
    y_true = test_labels

    cv_results = knn_clf.cv_results_
    accuracy = cv_results['mean_test_score']
    print('--------------------- RESULTS FROM GRIDSEARCH --------------------- \n', cv_results)
    print('--------------------- BEST PARAMETERS FROM GRIDSEARCH --------------------- \n', knn_clf.best_params_)
    print(' OVERALL ACCURACY:', np.round(np.sum(accuracy)/len(accuracy)*100,2))    

    plt.figure(1)
    plt.plot(accuracy)
    plt.xlabel('Fold')
    plt.ylabel('Mean accuracy of test score')
    plt.show()

    metrics = compute_metrics(y_true, y_pred)

    return metrics


def EEGNet_classifier(train_data, test_data, val_data, train_labels, test_labels, val_labels, epoch_duration):

    # configure the EEGNet-8,2,16 model with kernel length of 32 samples (other 
    # model configurations may do better, but this is a good starting point)
    model = EEGNet(nb_classes=2, Chans=var.NUM_CHANNELS, Samples=(epoch_duration*var.SFREQ)+1, 
                   dropoutRate = 0.5, kernLength = 64, F1 = 8, D = 2, F2 = 16, 
                   dropoutType = 'Dropout')
    
    # compile the model and set the optimizers
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', 
                  metrics = ['accuracy'])
    
    # count number of parameters in the model
    numParams    = model.count_params() 
    

    # set a valid path for your system to record model checkpoints
    checkpointer = ModelCheckpoint(filepath='/tmp/checkpoint.h5', verbose=1,
                               save_best_only=True)
    
    ###############################################################################
    # if the classification task was imbalanced (significantly more trials in one
    # class versus the others) you can assign a weight to each class during 
    # optimization to balance it out. This data is approximately balanced so we 
    # don't need to do this, but is shown here for illustration/completeness. 
    ###############################################################################

    # the syntax is {class_1:weight_1, class_2:weight_2,...}. Here just setting
    # the weights all to be 1
    class_weights = {0:1, 1:1}

    ################################################################################
    # fit the model. Due to very small sample sizes this can get
    # pretty noisy run-to-run, but most runs should be comparable to xDAWN + 
    # Riemannian geometry classification (below)
    ################################################################################
    fittedModel = model.fit(train_data, train_labels, batch_size = 64, epochs = 300, 
                            verbose = 2, validation_data=(val_data, val_labels),
                            callbacks=[checkpointer], class_weight = class_weights)

    # load optimal weights
    model.load_weights('/tmp/checkpoint.h5')

    ###############################################################################
    # can alternatively used the weights provided in the repo. If so it should get
    # you 93% accuracy. Change the WEIGHTS_PATH variable to wherever it is on your
    # system.
    ###############################################################################

    # WEIGHTS_PATH = /path/to/EEGNet-8-2-weights.h5 
    # model.load_weights(WEIGHTS_PATH)

    ###############################################################################
    # make prediction on test set.
    ###############################################################################

    probs       = model.predict(test_data)
    preds       = probs.argmax(axis = -1)  
    acc         = np.mean(preds == test_labels)
    print("Classification accuracy: %f " % (acc))

    names        = ['Not stressed', 'Stressed']
    plt.figure(0)
    plot_confusion_matrix(preds, test_labels, names, title = 'EEGNet-8,2')
    return probs

