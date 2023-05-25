from utils.data_processing import extract_eeg_data, segment_data, create_train_test_split
from utils.labels import get_stai_labels, get_pss_labels
from utils.valid_recordings import get_valid_recordings
import numpy as np

def prepare_data(data_type, label_type, epoch_duration=5):
    '''
    Prepare data of different types for classification.

    Parameters
    ----------
    data_type : str
        String containing the data type to be classified. Either 'raw', 'filtered', or 'decomposed'
    label_type : str
        Input can be either 'STAI' or 'PSS', to indicate either STAI-labels or PSS-labels.
    epoch_duration : int
        The chosen epoch duration for segmenting the data, in seconds. Default is 5 seconds.
    
    
    Returns
    -------
    train_data : nd.array
        An array containing the training data of the given data type
    train_labels : nd.array
        An array containing the training labels of the given data type
    test_data : nd.array
        An array containing the test data of the given data type
    test_labels : nd.array
        An array containing the test labels of the given data type

    '''
    valid_recordings = get_valid_recordings(data_type=data_type)
    data = extract_eeg_data(valid_recordings=valid_recordings, data_type=data_type)

    if label_type == 'STAI':
        labels = get_stai_labels(valid_recordings, cutoff=40)
    elif label_type == 'PSS':
        labels = get_pss_labels(valid_recordings=valid_recordings, threshold=4)
    else:
        print('Suggested label_type is not supported.')

    segmented_data, segmented_labels = segment_data(data, labels, epoch_duration)

    train_data, train_labels, test_data, test_labels = create_train_test_split(segmented_data, segmented_labels, epoch_duration=epoch_duration)

    print(f"Shape of train data set: {train_data.shape}")
    print(f"Shape of train labels set: {train_labels.shape}")

    print(f"Shape of test data set: {test_data.shape}")
    print(f"Shape of test labels set: {test_labels.shape}")

    return train_data, train_labels, test_data, test_labels
    





        