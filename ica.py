import scipy
import os
import numpy as np
import mne
import matplotlib.pyplot as plt

class Recording:
    #Data paths
    root_path = 'C:/Users/idamand/Documents/Master2'
    dir_raw = root_path + 'Data/Raw_eeg'
    dir_filtered = root_path + 'Data/Filtered' #need to change, also in function save_data

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

    def save_data(self):
        title = f"/sub-{self.sub_nr}_ses-{self.ses_nr}_run-{self.run_nr}"
        clean_data = self.reconst_arr.to_data_frame(scaling=1e6)
        clean_data = clean_data.to_numpy()
        clean_data = np.transpose(clean_data)
        clean_dict = {"Clean_data" : clean_data[1:, :]}
        scipy.io.savemat(f"{self.root_path}/Data/ICA_data/{title}.mat", clean_dict)

    def init_filter(self):
        band_pass = self.raw_array.copy().filter(1, 50) #check this out
        sav_gol = band_pass.copy().savgol_filter(h_freq=10, verbose=False)
        return sav_gol

    def init_ICA(self):
        self.ica = mne.preprocessing.ICA(n_components=8, max_iter=10000, random_state=97)
        self.ica.fit(self.filt_arr)
        
    def plot_sources(self):
        self.ica.plot_sources(self.filt_arr, title=f'ICA components sub-{self.sub_nr}_ses-{self.ses_nr}_run-{self.run_nr}', show_scrollbars=False)
        self.ica.plot_components(colorbar=True, reject='auto')
        
    def plot_properties(self, components):
        self.ica.plot_properties(self.filt_arr, picks = components)
        
    def test_exclude(self, components):
        self.ica.plot_overlay(self.filt_arr, exclude=components, picks='eeg', stop = 300.)
        #self.ica.plot_overlay(self.filt_arr, exclude=components, picks='eeg', show = True)
 

    def exclude_ICA(self, components):
        self.ica.exclude = components
        self.reconst_arr = self.filt_arr.copy()
        self.ica.apply(self.reconst_arr)
    
    def plot(self, data_type, save=False):
        if data_type == 'ica' and save == True:
            with mne.viz.use_browser_backend('matplotlib'):
                title = f"ICA components sub-{self.sub_nr}_ses-{self.ses_nr}_run-{self.run_nr}"
                fig = self.ica.plot_sources(self.filt_arr, title=title, 
                                            show_scrollbars=False)
                fig.savefig(f'{self.root}/Figures/{title}.png') 

        else:
            if data_type == 'raw':
                data = self.raw_arr
                title = f"Raw data sub-{self.sub_nr}_ses-{self.ses_nr}_run-{self.run_nr}"
            elif data_type == 'filtered': 
                data = self.filt_arr
                title = f"Filtered data sub-{self.sub_nr}_ses-{self.ses_nr}_run-{self.run_nr}"
            elif data_type == 'reconstructed':
                data = self.reconst_arr
                title = f"Reconstructed data sub-{self.sub_nr}_ses-{self.ses_nr}_run-{self.run_nr}"
                pass

            if not save:
                data.plot(duration = 25, title=title, n_channels=self.n_channels, scalings=None, show_scrollbars=False)
            else:
                with mne.viz.use_browser_backend('matplotlib'):
                    #scalings = 18 is good
                    fig = data.plot(duration = 600, title=f'{title}', n_channels=4, scalings=None, show_scrollbars=False)
                    fig.savefig(f'{self.root}/Figures/{title}.png') 


