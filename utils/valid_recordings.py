import os
import logging
import utils.variables as var
from utils.data_processing import read_mat_data

def generate_all_recs():
    """
    Generate all possible recording names based on the number of participants, sessions, and runs.

    Returns
    -------
    recs : list of str
        List of all possible recording names in the format 'P00X_S00X_00X'.

    """

    recs = []
    for i in range(1, var.NUM_SUBJECTS + 1):
        for j in range(1, var.NUM_SESSIONS + 1):
            for k in range(1, var.NUM_RUNS + 1):
                subject = f'P{str(i).zfill(3)}'
                session = f'S{str(j).zfill(3)}'
                run = f'{str(k).zfill(3)}'
                rec = f'{subject}_{session}_{run}'
                recs.append(rec)
    return recs

def filter_valid_recs(recs):
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

    dir = var.DIR_RAW
    valid_recs = []
    for rec in recs:
        subject, session, run = rec.split('_')
        f_name = os.path.join(
            dir, f'sub-{subject}_ses-{session}_run-{run}.mat')
        try:
            data = read_mat_data(f_name)
            if data.n_times / data.info['sfreq'] >= 4.30 * 60:
                valid_recs.append(rec)
        except:
            logging.error(f"Failed to read data for recording {rec}")
            continue
    return valid_recs


def get_valid_recs():
    """
    This function returns a list of valid recording names based on the raw EEG data.

    Returns
    -------
    valid_recs : list
        A list of valid recording names in the format 'P00X_S00X_00X'.
    """

    recs = generate_all_recs()
    valid_recs = filter_valid_recs(recs)
    return valid_recs