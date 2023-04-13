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

    #n_channels = var.NUM_CHANNELS
    #features_per_channel = 2

    #features = []
    #for epoch in data:
    #    n_trials           = len(epoch)
    #    features_for_epoch = np.empty([n_trials, n_channels * features_per_channel])
        
    #    for channel, data in enumerate(epoch):
    #       trial = epoch[data]
    #        mobility_spect       = mnf.compute_hjorth_mobility_spect(var.SFREQ, trial)
    #        complexity_spect     = mnf.compute_hjorth_complexity_spect(var.SFREQ, trial)
    #        features_for_epoch[channel] = np.concatenate([mobility_spect, complexity_spect])

    #    features_for_epoch = features_for_epoch.reshape([n_trials, n_channels*features_per_channel])
    #    features.append(features_for_epoch)

    features_per_channel = 2
    n_epoch, n_channels, n_samples = data.shape 
    features = np.empty([n_epoch, n_channels * features_per_channel])  
    for i, epochs in enumerate(data):
        for j, samples in enumerate(epochs):
            mobility_spect       = mnf.compute_hjorth_mobility_spect(var.SFREQ, samples)
            complexity_spect     = mnf.compute_hjorth_complexity_spect(var.SFREQ, samples)
            print('Mob spect: ', mobility_spect)
            features[i][j] = np.concatenate([mobility_spect, complexity_spect])
    features = features.reshape([n_epoch, n_channels * features_per_channel])
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

    first_key = next(iter(data[0]))
    n_channels, _ = data[0][first_key].shape
    features_per_channel = 4

    features = []
    for fold in data:
        n_trials          = len(fold)
        features_for_fold = np.empty([n_trials, n_channels * features_per_channel])

        for j, key in enumerate(fold):
            trial = fold[key]
            app_entropy          = mnf.compute_app_entropy(trial)
            samp_entropy         = mnf.compute_samp_entropy(trial)
            spect_entropy        = mnf.compute_spect_entropy(var.SFREQ, trial)
            svd_entropy          = mnf.compute_svd_entropy(trial)
            features_for_fold[j] = np.concatenate([app_entropy, samp_entropy, spect_entropy, svd_entropy])

        features_for_fold = features_for_fold.reshape([n_trials, n_channels*features_per_channel])
        features.append(features_for_fold)
    return features