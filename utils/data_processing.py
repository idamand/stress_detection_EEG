import mne
import os
import numpy as np
import scipy
import logging
import utils.variables as var
from sklearn.model_selection import StratifiedKFold, train_test_split
import scipy.io as sio

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
    return scipy.io.loadmat(filename)


def read_eeg_data(data_type, filename):
    if data_type == 'raw':
        data_key = 'raw_eeg_data'
    elif data_type == 'ssp':
        data_key = 'Clean_data'
    elif data_type == 'decomp':
        data_key = '"Decomp_data"'
    else:
        print(f'No data with data_type = {data_type} found')
        return 0

    data = sio.loadmat(filename)[data_key]
    data = data[:,:75000] # 5 MIN * 60 SEK/MIN * 250 SAMPLES/SEK = 75 000 SAMPLES

    info = mne.create_info(8, sfreq=var.SFREQ, ch_types= 'eeg', verbose=None)
    raw_array = mne.io.RawArray(data, info)
    mapping = {'0':'F4','1':'Fp2','2':'C3','3':'FC6','4':'O1','5':'Oz','6':'FT9','7':'T8'}
    mne.rename_channels(raw_array.info, mapping)

    return raw_array

def extract_eeg_data(valid_recordings, data_type):
    '''
    Construct dictionairies with EEG data.
    '''

    assert(data_type in var.DATA_TYPES)

    if data_type == 'raw':
        dir         = var.DIR_RAW
        data_key    = 'Data'
    elif data_type == 'ssp':
        dir         = var.DIR_SSP
        data_key    = 'Clean data'
    elif data_type == 'decomp':
        dir         = var.DIR_DECOMP
        data_key    = 'Decomp data'
    else:
        print('No files matching data type found')
        return 0

    eeg_data = {}
    for recording in valid_recordings:
        subject, session, run = recording.split('_') #splits with regards to underscore _
        f_name = os.path.join(dir, f'sub-{subject}_ses-{session}_run-{run}.mat')

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
    return: dict
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

def create_train_test_split(data, labels, epoch_duration):
    '''
    Split data and labels into train and test sets to be used for k-fold cross validation. The split is done across subject to ensure no subject is in both train or test sets.

    Parameters
    ----------
    data : dict
        Dictionary containing the epoched data
    labels : dict
        Dictionary containing the epoched labels

    Returns
    -------
    train_data : numpy.ndarray
        An array with shape `(n_subjects, n_sessions * n_runs)` containing
        labels. Each row corresponds to a subject, and each column
        corresponds to a session/run. The values in the array are integers
        representing the label (0, 1, or 2) for that subject, session, and
        run.
    '''

    keys_list = list(data.keys())
    
    #Creates a list of all subjects
    subject_list = []
    for i in range(var.NUM_SUBJECTS+1):
        subject = f'P{str(i).zfill(3)}'
        for key in keys_list:
            if subject in key and subject not in subject_list:
                subject_list.append(subject)

    mean_labels_list = []
    for i in range(var.NUM_SUBJECTS+1):
        sum_label = 0
        num_recordings = 0
        subject = f'P{str(i).zfill(3)}'
        for key, value in labels.items():
            if subject in key:
                sum_label += value
                num_recordings += 1
        if num_recordings == 0:
            continue
        else:
            mean_label = sum_label/num_recordings
            mean_labels_list.append(round(mean_label,0))

    subjects_train, subjects_test, mean_labels_train, mean_labels_test = train_test_split(subject_list, mean_labels_list,
                                                                                          test_size=0.2, random_state=42,
                                                                                          stratify=mean_labels_list)

    for subject in subjects_train:
        if subject in subjects_test:
            print(f'ERROR: Subject {subject} in both training and test list') 
   
    print('Subjects test: ', subjects_test, '\n Subjects train: ', subjects_train)
    
    train_data_dict, train_labels_dict = reconstruct_dicts(subjects_train, data, labels)
    test_data_dict, test_labels_dict   = reconstruct_dicts(subjects_test, data, labels)

    train_data      = dict_to_arr(train_data_dict, epoch_duration)
    test_data       = dict_to_arr(test_data_dict, epoch_duration)

    train_labels    = np.reshape(np.array(list(train_labels_dict.values())), (len(train_data_dict),1))
    test_labels     = np.reshape(np.array(list(test_labels_dict.values())), (len(test_data_dict),1))

    train_labels    = train_labels.ravel()
    test_labels     = test_labels.ravel()

    return train_data, train_labels, test_data, test_labels

def reconstruct_dicts(subjects_list, x_dict, y_dict):
    '''
    Reconstructs the dictionarys after the dataset has been split into train-, validation- and test-sets
    '''
    data_dict = {}
    labels_dict = {}

    for subject in subjects_list:
        # Reconstructing data dict
        for key, val in x_dict.items():
            if subject in key:
                data_dict[key] = val

        #Reconstructing labels dict
        for key, val in y_dict.items():
            if subject in key:
                labels_dict[key] = val

    return(data_dict, labels_dict)
    
def dict_to_arr(data_dict, epoch_duration):
    '''
    Turns dictionary into numpy array
    '''
    keys_list = list(data_dict.keys())
    
    data_arr = np.empty((len(keys_list), var.NUM_CHANNELS, epoch_duration*var.SFREQ+1))
    
    i = 0
    for key in keys_list:
        data = data_dict[key]
        data_arr[i] = data
        i += 1

    return data_arr