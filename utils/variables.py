import os

DIR_RAW = 'Data/Raw_eeg'
DIR_FILTERED = 'Data/SSP_eeg'
DIR_PSD = 'Data/psd'

LABEL_PATH = 'Data/STAI_grading.xls'



NUM_SUBJECTS = 28
NUM_SESSIONS = 2
NUM_RUNS = 2

NUM_CHANNELS = 8
SFREQ = 250
MAPPING = {'0':'F4','1':'Fp2','2':'C3','3':'FC6','4':'O1','5':'Oz','6':'FT9','7':'T8'}

NUM_SAMPLES = 75000
EPOCH_LENGTH = 5

DATA_TYPES = ['raw', 'ssp', 'psd']