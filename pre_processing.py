import numpy as np
import mne
import utils.variables as var
from mne.filter import construct_iir_filter, filter_data, create_filter


#Implement SSP, filtering etc.


def create_raw_array(array):
    '''
    Creates a RawArray object

    Parameters
    ----------
    array : ndarray
        The array that will be converted into a RawArray

    Returns
    -------
    raw_array : RawArray
        An instance of a Raw object. Includes the data and info specified for the project. 

    '''
    info = mne.create_info(8, sfreq=var.SFREQ, ch_types='eeg', verbose=None)
    raw_array = mne.io.RawArray(array, info)
    mapping = {'0':'F4','1':'Fp2','2':'C3','3':'FC6','4':'O1','5':'Oz','6':'FT9','7':'T8'}
    mne.rename_channels(raw_array.info, mapping)

    return raw_array

def init_filter(data):
    '''
    Initial filtering of the data. Includes a bandpass filter between 1-50 Hz, and a Savitzky-Golay filter.

    Parameters
    ----------
    data : dict
        Dictionary of the data. Shape (keys, number of channels, number of samples)
    
    Returns
    -------
    filtered data : dict
        A dictionary containing the filtered data

    '''
    
    filtered_data = {}
    for key in data.keys():
        data_for_filtering = data[key].copy().get_data()
        raw_array = create_raw_array(data_for_filtering)
        band_pass = raw_array.filter(1, 50)
        sav_gol = band_pass.copy().savgol_filter(h_freq=10, verbose=False)
        filtered_data[key] = sav_gol
    return filtered_data

def filter_EEG(data, low_freq, high_freq, method='iir'):
    '''
    A function that filters the data, with a either bandpass, lowpass or highpass filters using the MNE function filter_data(..)

    Parameters
    ----------
    data : dict
        A dictionary containing the data. Shape (keys, number of channels, number of samples)
    l_freq : int
        The lower cutoff frequency given in Hz. If None then the filter will be low-pass.
    h_freq : int
        The upper cutoff frequency given in Hz. If None then the filter will be highpass. 

    Returns
    -------
    filtered_data : dict
        A dictionary with the filtered data with shape (keys, number of channels, number of samples)
    '''
    sfreq = var.SFREQ
    filtered_data = {}
    for key in data.keys():
        data_for_filtering = data[key].copy().get_data()
        filtering = filter_data(data_for_filtering, sfreq=sfreq, l_freq=low_freq, h_freq=high_freq, method=method)
        filtered_raw_array = create_raw_array(filtering)
        filtered_data[key]=filtered_raw_array
    return filtered_data

def decompose_EEG(data, decomp_type='gamma'):
    '''
    
    '''
    if decomp_type == 'gamma':
        low_freq    = 30
        high_freq   = None
    elif decomp_type == 'beta':
        low_freq    = 13
        high_freq   = 30
    elif decomp_type == 'alpha':
        low_freq    = 8
        high_freq   = 13

    init_filtering = init_filter(data)
    decomposition = filter_EEG(init_filtering, low_freq=low_freq, high_freq=high_freq)
    
    return decomposition

