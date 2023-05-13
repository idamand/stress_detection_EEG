import os
import logging
import utils.variables as var
from utils.data_processing import read_mat_data, read_eeg_data

def generate_all_recordings():
    """
    Generate all possible recording names based on the number of participants, sessions, and runs.

    Returns
    -------
    recs : list of str
        List of all possible recording names in the format 'P00X_S00X_00X'.

    """

    recordings = []
    for i in range(1, var.NUM_SUBJECTS + 1):
        for j in range(1, var.NUM_SESSIONS + 1):
            for k in range(1, var.NUM_RUNS + 1):
                subject = f'P{str(i).zfill(3)}'
                session = f'S{str(j).zfill(3)}'
                run = f'{str(k).zfill(3)}'
                rec = f'{subject}_{session}_{run}'
                recordings.append(rec)
    return recordings

def filter_valid_recordings(recordings, data_type):
    """
    This function returns the valid recordings from a list of recordings.

    Parameters
    ----------
    recs : list
        A list of recording names in the format 'P00X_S00X_00X'.

    Returns
    -------
    valid_recs : list
        A list of valid recordings in the format 'P00X_S00X_00X'.
    """
    if data_type == 'raw':
        dir = var.DIR_RAW
    elif data_type == 'ssp':
        dir = var.DIR_SSP
    elif data_type == 'decomp':
        dir = var.DIR_DECOMP
    else:
        print('No data with datatype = {data_type} found')
    
    valid_recordings = []
    for recording in recordings:
        subject, session, run = recording.split('_')
        f_name = os.path.join(
            dir, f'sub-{subject}_ses-{session}_run-{run}.mat')
        try:
            data = read_eeg_data(data_type, f_name)
            if data.n_times / data.info['sfreq'] >= 4.30 * 60:
                valid_recordings.append(recording)
        except:
            logging.error(f"Failed to read data for recording {recording}")
            continue
    return valid_recordings


def get_valid_recordings(data_type):
    """
    This function returns a list of valid recording names based on the raw EEG data.

    Returns
    -------
    valid_recs : list
        A list of valid recording names in the format 'P00X_S00X_00X'.
    """

    recordings = generate_all_recordings()
    valid_recordings = filter_valid_recordings(recordings, data_type)
    return valid_recordings
