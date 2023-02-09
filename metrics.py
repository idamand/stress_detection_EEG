import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix

def compute_metrics(y_true, y_pred):
    '''
    Compute the confusion matrix of the input, accuracy, sensitivity, and specificity. 
    Input is y_true and y_predicted. 
    Output is display of confusion matrix and an array of accuracy, sensitivity, and specificity. 
    '''

    confusion_matrix_display = ConfusionMatrixDisplay.from_predictions(y_true, y_pred)
    plt.show()

    conf_matrix = confusion_matrix(y_true, y_pred)

    true_positive = conf_matrix[0, 0]
    true_negative = conf_matrix[1, 1]
    false_positive = conf_matrix[1, 0]
    false_negative = conf_matrix[0, 1]

    accuracy = (true_positive+true_negative)/(true_positive + true_negative + false_positive + false_negative)

    sensitivity = true_positive/(true_positive + false_negative)
    specificity = true_negative/(false_positive + true_negative)

    metrics = np.array([accuracy, sensitivity, specificity])

    return metrics





