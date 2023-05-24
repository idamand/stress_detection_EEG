import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix
from matplotlib.colors import LinearSegmentedColormap

def compute_metrics(y_true, y_pred):
    '''
    Compute the confusion matrix of the input, accuracy, sensitivity, and specificity. 
    Input is y_true and y_predicted. 
    Output is display of confusion matrix and an array of accuracy, sensitivity, and specificity. 
    '''
    conf_matrix = confusion_matrix(y_true, y_pred)

    true_negative   = conf_matrix[0,0]
    false_negative  = conf_matrix[1,0]
    true_positive   = conf_matrix[1,1]
    false_positive  = conf_matrix[0,1]

    accuracy = ((true_positive+true_negative)/(true_positive + true_negative + false_positive + false_negative)) *100

    sensitivity = (true_positive/(true_positive + false_negative)) *100
    specificity = (true_negative/(false_positive + true_negative)) *100

    labels = ('Not stressed', 'Stressed')
    colors = ["#ef8114", "#b01b81", "#482776"]
    cmap1 = LinearSegmentedColormap.from_list("mycmap", colors)
    confusion_matrix_display = ConfusionMatrixDisplay.from_predictions(y_true, y_pred, cmap=cmap1, display_labels=labels)
    
    textstr = f"Accuracy: {round(accuracy,2)}% \nSensitivity: {round(sensitivity,2)}% \nSpecificity: {round(specificity,2)}%"
    props = dict(boxstyle='round', facecolor='white', alpha=0.5)
    xpos = 0.5
    ypos = -0.2
    confusion_matrix_display.ax_.text(xpos, ypos, textstr, transform=confusion_matrix_display.ax_.transAxes, fontsize=16,
            verticalalignment='top', bbox=props, ha='center')
    
    plt.show

    metrics = np.array([accuracy, sensitivity, specificity])

    return metrics





