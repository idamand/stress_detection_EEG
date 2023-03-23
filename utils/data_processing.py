import mne
import os
import numpy as np
import scipy
import logging
import utils.variables as var
from sklearn.model_selection import StratifiedKFold

def read_mat_data(filename):
    """
    Read .mat data from file.

    Parameters
    ----------
    filename : str
        Path to the file to be read.

    Returns
    -------
    data : instance of Raw object
        The EEG data contained in the file.
    """
    return sio.loadmat(filename)


def read_eeg_data(data_type, filename):
    if data_type == 'raw':
        data_key = 'raw_eeg_data'
    elif data_type == 'filtered':
        data_key = 'Clean_data'
    else:
        print(f'No data with data_type = {data_type} found')
        return 0

    data = scipy.io.loadmat(filename)[data_key]
    info = mne.create_info(8, sfreq=var.SFREQ, ch_types= 'eeg', verbose=None)
    raw_array = mne.io.RawArray(data, info)
    mapping = {'0':'F4','1':'Fp2','2':'C3','3':'FC6','4':'O1','5':'Oz','6':'FT9','7':'T8'}
    mne.rename_channels(raw_array.info, mapping)

    return raw_array

def extract_eeg_data(valid_recordings, data_type):
    '''
    
    '''

    assert(data_type in var.DATA_TYPES)

    if data_type == 'raw':
        dir         = var.DIR_RAW
        data_key    = 'Data'
    elif data_type == 'filtered':
        dir         = var.DIR_FILTERED
        data_key    = 'Clean data'
    else:
        print('No files matching data type found')
        return 0

    eeg_data = {}
    for recording in valid_recordings:
        subject, session, run = recording.split('_') #splits with regards to underscore _
        f_name = os.path.join(dir, f'sub-{subject}_ses{session}_run-{run}.mat')

        try:
            data = read_eeg_data(data_type, f_name)
        except:
            logging.error(f"2) Failed to read data for recording {recording}")
            data = None

        key = f'{subject}_{session}_{run}'
        eeg_data[key] = data
    return eeg_data

def segment_data(x_dict, y_dict, epoch_duration=5):
    '''
    
    '''
    overlap_duration = 0.0

    x_epochs = {}
    y_epochs = {}

    
    for key, raw_array in x_dict.items():
        events = mne.make_fixed_length_events(raw_array, 
                                              stop=5*60, 
                                              duration=epoch_duration, 
                                              overlap=overlap_duration)

        epochs = mne.Epochs(raw_array, 
                            events, tmin=0, 
                            tmax=epoch_duration, 
                            baseline=None, 
                            preload=True)
        
        for i, epoch in enumerate(epochs):
            x_epochs[f'{key}_epoch{i}'] = epoch
            y_epochs[f'{key}_epoch{i}'] = y_dict[key]

    return x_epochs, y_epochs

def stratified_kfold_split(x_epochs, y_epochs, n_splits=5, shuffle=True, random_state=None):
    """
    Perform k-fold cross-validation on a dataset of EEG epochs.
    Parameters
    ----------
    x_epochs : dict
        A dictionary of EEG epochs, where each key is an epoch ID and each value is a numpy array of shape (n_channels, n_samples).
    y_epochs : dict
        A dictionary of target labels for each epoch, where each key is an epoch ID and each value is an integer target label.
    n_splits : int, optional
        The number of folds to create in the cross-validation. Default is 5.
    shuffle : bool, optional
        Whether to shuffle the data before creating folds. Default is True.
    random_state : int or RandomState, optional
        If an integer, `random_state` is the seed used by the random number generator. If a RandomState instance, `random_state` is the random number generator. If None, the random number generator is the RandomState instance used by `np.random`. Default is None.
    Returns
    -------
    train_epochs : list of dicts
        A list of training epochs for each fold, where each dictionary is of the same format as `x_epochs`.
    test_epochs : list of dicts
        A list of test epochs for each fold, where each dictionary is of the same format as `x_epochs`.
    train_labels : list of dicts
        A list of target labels for the training epochs for each fold, where each dictionary is of the same format as `y_epochs`.
    test_labels : list of dicts
        A list of target labels for the test epochs for each fold, where each dictionary is of the same format as `y_epochs`.
    """

    subject_ids = np.unique(['_'.join(k.split('_')[:-1]) for k in x_epochs.keys()])
    y_subjects = np.array([y_epochs[f'{subject_id}_epoch0'] for subject_id in subject_ids])

    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

    train_epochs = []
    train_labels = []
    test_epochs = []
    test_labels = []
    for train_subjects, test_subjects in skf.split(subject_ids, y_subjects):
        train_subjects_list = subject_ids[train_subjects]
        test_subjects_list = subject_ids[test_subjects]
        train_epochs.append({k: v for k, v in x_epochs.items() if '_'.join(k.split('_')[:-1]) in train_subjects_list})
        train_labels.append({k: v for k, v in y_epochs.items() if '_'.join(k.split('_')[:-1]) in train_subjects_list})
        test_epochs.append({k: v for k, v in x_epochs.items() if '_'.join(k.split('_')[:-1]) in test_subjects_list})
        test_labels.append({k: v for k, v in y_epochs.items() if '_'.join(k.split('_')[:-1]) in test_subjects_list})

    return train_epochs, test_epochs, train_labels, test_labels
