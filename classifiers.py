from matplotlib import pyplot as plt
import numpy as np
#from sklearn.metrics import plot_confusion_matrix

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from metrics import compute_metrics
from sklearn.model_selection import GridSearchCV

from EEGNet.EEGModels import EEGNet
from tensorflow.keras import utils as np_utils
from tensorflow.keras.callbacks import ModelCheckpoint
#from keras.utils import np_utils
#from keras.callbacks import ModelCheckpoint
from pyriemann.utils.viz import plot_confusion_matrix

import utils.variables as var
import mne


def SVM(train_data, test_data, train_labels, test_labels):
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
        'C': [0.1, 1, 10, 100, 1000],
        'kernel': ['rbf', 'poly']
    }

    #weights = {0:67, 1:33}

    svm_clf = GridSearchCV(SVC(), param_grid=param_grid, refit=True)
    svm_clf.fit(train_data, train_labels)

    y_pred = svm_clf.predict(test_data)
    y_true = test_labels

    metrics = compute_metrics(y_true, y_pred)

    return metrics


def RF(data, labels):
    '''
    Input: data of shape (n_samples, n_features), and labels of shape (n_samples). Performs random forest classification
    '''

    '''
    n_estimators = 10
    max_depth = 5
    weights = {0:67, 1:33}
    
    rf_clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, class_weight=weights, random_state=42)
    rf_clf.fit(x_train, y_train)

    y_pred = rf_clf.predict(x_test)
    y_true = y_test

    metrics = compute_metrics(y_true, y_pred)
    '''

    return 0


def EEGNet_classifier(train_data, test_data, val_data, train_labels, test_labels, val_labels):

    # configure the EEGNet-8,2,16 model with kernel length of 32 samples (other 
    # model configurations may do better, but this is a good starting point)
    model = EEGNet(nb_classes=2, Chans=var.NUM_CHANNELS, Samples=(var.EPOCH_LENGTH*var.SFREQ)+1, 
                   dropoutRate = 0.5, kernLength = 32, F1 = 8, D = 2, F2 = 16, 
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

