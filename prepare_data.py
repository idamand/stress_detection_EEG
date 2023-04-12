from utils.data_processing import extract_eeg_data, segment_data, train_test_val_split, dict_to_arr
from utils.labels import get_stai_labels, compute_stai_scores
from utils.valid_recordings import get_valid_recordings

def prepare_data(data_type):
    '''
    Prepare data of different types for classification.

    Parameters
    ----------
    data_type : str
        String containing the data type to be classified. Either 'raw' or 'filtered'
    
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
    val_data : nd.array
        An array containing the validation data of the given data type
    val_labels : nd.array
        An array containing the validation labels of the given data type
    '''

    if data_type == 'raw':

        valid_recordings = get_valid_recordings('raw')

        labels = get_stai_labels(valid_recordings, cutoff=40)
        data = extract_eeg_data(valid_recordings=valid_recordings, data_type='raw')

        segmented_data, segmented_labels = segment_data(data, labels, epoch_duration=5)

        