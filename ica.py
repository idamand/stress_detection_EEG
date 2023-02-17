import scipy
import os
import numpy as np
import mne
import matplotlib.pyplot as plt

class Recording:
    #Data paths
    dir_raw = ''
    dir_filtered = ''

    #Parameters
    Fs = 250
    ch_type = 'eeg'
    n_channels = 8

    def __init__(self, sub_nr, ses_nr, run_nr):
        self.sub_nr = sub_nr
        self.ses_nr = ses_nr
        self.run_nr = run_nr

        #Load data
        self.load_data()
        
        #Create MNE rawArray and print: mne.create_info(ch_names, sfreq, ch_types='misc', verbose=None)
        info = mne.create_info(8, sfreq=self.Fs, ch_types=self.ch_type, verbose=None)
        self.raw_array = mne.io.RawArray(self.data, info)

        mapping = {'0':'F4','1':'Fp2','2':'C3','3':'FC6','4':'O1','5':'Oz','6':'FT9','7':'T8'}
        mne.rename_channels(self.raw_array.info, mapping)

        #Initial filtering
        self.filt_array = self.init_filter()

        #Set montage
        montage = mne.channels.make_standard_montage('standard_1020')
        self.filt_array.set_montage(montage)  

    #__________________________________ Functions ______________________________________

    def load_data(self):
        dir = self.dir_raw
        data_key = 'raw_eeg_data'

        #Load one recording
        filename = f"/sub-{self.sub_nr}_ses-{self.ses_nr}_run-{self.run_nr}.mat"
        f = dir + filename
        self.data = scipy.io.loadmat(f)[data_key]



