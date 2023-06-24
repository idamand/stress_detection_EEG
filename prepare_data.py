from utils.data_processing import extract_eeg_data, segment_data, create_train_test_split, dict_to_arr_labels, dict_to_arr
from utils.labels import get_stai_labels, get_pss_labels
from utils.valid_recordings import get_valid_recordings
from features import time_series_features, entropy_features, hjorth_features, psd_features
import numpy as np

def prepare_data(data_type, label_type, epoch_duration=5, feature_type='None'):
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
    feature_type : str
        The chosen feature type, default is None, available feature_types is 'time_series', 'entropy', 'hjorth', or 'psd'
    
    
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

    print(f'---------------- CHOSEN FEATURE TYPE: {feature_type}----------------')
    print(f"----- Shape of train data set: {train_data.shape} _________ Shape of train labels set: {train_labels.shape} -----")
    #print(f"Shape of train labels set: {train_labels.shape}")

    print(f"----- Shape of test data set:  {test_data.shape}  _________ Shape of test labels set:  {test_labels.shape}  -----")
    #print(f"Shape of test labels set: {test_labels.shape}")

    if feature_type == 'time_series':
        train_data_features = time_series_features(train_data)
        test_data_features  = time_series_features(test_data)
        print(f"Shape of train data features set: {train_data_features.shape}")
        print(f"Shape of test data features set: {test_data_features.shape}")
        return train_data_features, test_data_features, train_labels, test_labels

    elif feature_type == 'entropy':
        train_data_features = entropy_features(train_data)
        test_data_features  = entropy_features(test_data)
        print(f"Shape of train data features set: {train_data_features.shape}")
        print(f"Shape of test data features set: {test_data_features.shape}")
        return train_data_features, test_data_features, train_labels, test_labels

    elif feature_type == 'hjorth':
        train_data_features = hjorth_features(train_data)
        test_data_features  = hjorth_features(test_data)
        print(f"Shape of train data features set: {train_data_features.shape}")
        print(f"Shape of test data features set: {test_data_features.shape}")
        return train_data_features, test_data_features, train_labels, test_labels

    elif feature_type == 'psd':
        train_data_features = psd_features(train_data)
        test_data_features  = psd_features(test_data)
        print(f"Shape of train data features set: {train_data_features.shape}")
        print(f"Shape of test data features set: {test_data_features.shape}")
        return train_data_features, test_data_features, train_labels, test_labels

    elif feature_type == 'None':
        return train_data, test_data, train_labels, test_labels

    
    

def prepare_data_for_clustering(data_type, epoch_duration):
    '''
    Prepare data for cluster algorithms with time-series features. 
    '''

    valid_recordings = get_valid_recordings(data_type=data_type)
    data             = extract_eeg_data(valid_recordings=valid_recordings, data_type=data_type)

    
    stai_labels = get_stai_labels(valid_recordings, cutoff=40)
    pss_labels  = get_pss_labels(valid_recordings=valid_recordings, threshold=4)

    stai_labels_arr = dict_to_arr_labels(stai_labels)
    pss_labels_arr  = dict_to_arr_labels(pss_labels)

    segmented_data, _ = segment_data(data, stai_labels, epoch_duration) #arbitrary label type for segmenting

    data_arr        = dict_to_arr(data_dict=segmented_data, epoch_duration=epoch_duration)

    data_features   = time_series_features(data=data_arr)
    return data_features, stai_labels_arr, pss_labels_arr
   

    
    
    


        