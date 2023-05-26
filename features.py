import numpy as np
import mne_features.univariate as mnf
import utils.variables as var


def hjorth_features(data):
    '''
    Computes the features Hjorth mobility (spectral) and Hjorth complexity (spectral) using the package mne_features.

    Parameters
    ----------
    data : dict
        data (list of dicts): A list of dictionaries, where each dictionary contains EEG data for multiple trials. The keys in each dictionary represent trial IDs, and the values are numpy arrays of shape (n_channels, n_samples).
    
    Returns
    -------
    features : list of ndarrays
        list of ndarrays with the computed features.
    '''

    sfreq = var.SFREQ
    features_per_channel = 2
    n_recordings, n_channels, n_samples = data.shape

    features = np.empty([n_recordings, n_channels*features_per_channel])

    for i in range(n_recordings):
        samples          = data[i]
        mobility_spect   = mnf.compute_hjorth_mobility_spect(sfreq=sfreq, data=samples)
        complexity_spect = mnf.compute_hjorth_complexity_spect(sfreq=sfreq, data=samples)
        features[i]      = np.concatenate([mobility_spect, complexity_spect])

    features = features.reshape([n_recordings, n_channels*features_per_channel])
  
    return features

def entropy_features(data):
    '''
    Computes the features Approximate Entropy, Sample Entropy, Spectral Entropy and SVD entropy using the package mne_features.
    
    Parameters
    ----------
    data : dict
        A list of dictionaries, where each dictionary contains EEG data for multiple trials. The keys in each dictionary represent trial IDs, and the values are numpy arrays of shape (n_channels, n_samples).
    
    Returns
    -------
    features : list of ndarrays
        list of ndarrays of the computed features.
    '''

    sfreq = var.SFREQ
    features_per_channel = 4
    n_recordings, n_channels, n_samples = data.shape

    features = np.empty([n_recordings, n_channels*features_per_channel])

    for i in range(n_recordings):
        samples         = data[i]
        app_entropy     = mnf.compute_app_entropy(data=samples)
        samp_entropy    = mnf.compute_samp_entropy(data=samples)
        spect_entropy   = mnf.compute_spect_entropy(sfreq=sfreq, data=samples)
        svd_entropy     = mnf.compute_svd_entropy(data=samples)
        features[i]     = np.concatenate([app_entropy, samp_entropy, spect_entropy, svd_entropy])

    features = features.reshape([n_recordings, n_channels*features_per_channel])

    return features

def time_series_features(data):
    '''
    Compute time-series features from data.

    Parameters
    ----------
    data : ndarray, shape(n_recordings, n_channels, n_samples)
        An ndarray containing the data.
    
    Returns
    -------
    features : ndarray, shape(n_recordings, n_channels*features_per_channel)
        An ndarray of the computed features
    '''
    sfreq = var.SFREQ
    features_per_channel = 4
    n_recordings, n_channels, n_samples = data.shape

    features = np.empty([n_recordings, n_channels*features_per_channel])

    for i in range(n_recordings):
        samples     = data[i]
        ptp_amp     = mnf.compute_ptp_amp(data=samples)
        variance    = mnf.compute_variance(data=samples)
        rms         = mnf.compute_rms(data=samples)
        mean        = mnf.compute_mean(data=samples)
        features[i] = np.concatenate([ptp_amp, variance, rms, mean])

    features = features.reshape([n_recordings, n_channels*features_per_channel])

    return features

def psd_features(data):
    sfreq = var.SFREQ
    features_per_channel = 1
    n_recordings, n_channels, n_samples = data.shape

    features = np.empty([n_recordings, n_channels*features_per_channel])

    for i in range(n_recordings):
        samples = data[i]
        psd     = mnf.compute_pow_freq_bands(sfreq=sfreq, data=samples)
        features[i] = np.concatenate([psd])

    features = features.reshape([n_recordings, n_channels*features_per_channel])

    return features

def all_features(data):
    '''
    Compute hjorth, entropy and time-series features in one

    '''

    sfreq = var.SFREQ
    features_per_channel = 10
    n_recordings, n_channels, n_samples = data.shape

    features = np.empty([n_recordings, n_channels*features_per_channel])

    for i in range(n_recordings):
        samples             = data[i]
        mobility_spect      = mnf.compute_hjorth_mobility_spect(sfreq=sfreq, data=samples)
        complexity_spect    = mnf.compute_hjorth_complexity_spect(sfreq=sfreq, data=samples)
        app_entropy         = mnf.compute_app_entropy(data=samples)
        samp_entropy        = mnf.compute_samp_entropy(data=samples)
        spect_entropy       = mnf.compute_spect_entropy(sfreq=sfreq, data=samples)
        svd_entropy         = mnf.compute_svd_entropy(data=samples)
        ptp_amp             = mnf.compute_ptp_amp(data=samples)
        variance            = mnf.compute_variance(data=samples)
        rms                 = mnf.compute_rms(data=samples)
        mean                = mnf.compute_mean(data=samples)
        features[i]         = np.concatenate([mobility_spect, complexity_spect, app_entropy, samp_entropy, spect_entropy, svd_entropy, ptp_amp, variance, rms, mean])

    features = features.reshape([n_recordings, n_channels*features_per_channel])