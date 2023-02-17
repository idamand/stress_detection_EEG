import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from metrics import compute_metrics
from sklearn.model_selection import StratifiedKFold


def SVM(data, labels):
    '''
    Input: data of shape (n_samples, n_features) and labels of shape (n_samples). Performs support vector classification.
    '''
    #x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42) #TODO: change to stratified k_fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in skf.split(data, labels):
        x_train, x_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]

    c = 1.0
    kernel = 'rbf'
    weights = {0:67, 1:33}

    svm_clf = SVC(C=c, kernel=kernel, class_weight=weights)
    svm_clf.fit(x_train, y_train)

    y_pred = svm_clf.predict(x_test)
    y_true = y_test

    metrics = compute_metrics(y_true, y_pred)

    return metrics


def RF(data, labels):
    '''
    Input: data of shape (n_samples, n_features), and labels of shape (n_samples). Performs random forest classification
    '''
    #x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, random_state=42) #TODO: change to stratified k-fold

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    for train_index, test_index in skf.split(data, labels):
        x_train, x_test = data[train_index], data[test_index]
        y_train, y_test = labels[train_index], labels[test_index]


    n_estimators = 10
    max_depth = 5
    weights = {0:67, 1:33}
    
    rf_clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, class_weight=weights, random_state=42)
    rf_clf.fit(x_train, y_train)

    y_pred = rf_clf.predict(x_test)
    y_true = y_test

    metrics = compute_metrics(y_true, y_pred)

    return metrics


