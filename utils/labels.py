import numpy as np
import pandas as pd
import logging
import utils.variables as var
import math

def load_pss_labels(filename, threshold):
    '''
    Load and binarize PSS (Perceived Stress Scale) scores from an Excel file.

    Parameters
    ----------
    filename : str
        The path to the Excel file containing the PSS scores.
    threshold : int or float
        The threshold value to use for binarizing the PSS scores. Scores greater than `threshold` will be
        binarized to 1, and scores less than or equal to `threshold` will be binarized to 0.

    Returns
    -------
    scores : pandas.DataFrame
        A DataFrame containing the binarized PSS scores, where each row represents a subject and each
        column represents a session. Scores are binarized according to the `threshold` value.
    '''

    scores = pd.read_excel(filename, sheet_name='Rating 1-10', skiprows=[1])
    scores.iloc[:, 1:] = scores.iloc[:, 1:].applymap(
        lambda x: x if pd.isna(x) else x > threshold)
    
    return scores

def filter_pss_labels(scores, valid_recs):
    """
    Filter PSS (Perceived Stress Scale) scores by record ID.

    Parameters
    ----------
    scores : pandas.DataFrame
        A DataFrame containing PSS scores for each subject and session.
    valid_recs : list of str
        A list of valid record IDs to keep in the filtered scores.

    Returns
    -------
    filtered_labels : dict
        A dictionary containing the filtered PSS scores, where the keys are record IDs and the values are the corresponding scores.
    """
    scores_dict = {}
    for i, _ in scores.iterrows():
        for j in range(len(scores.columns)):
            session_no = j // 2 + 1
            run_no = j % 2 + 1
            subject = f'P{str(i+1).zfill(3)}'
            session = f'S{str(session_no).zfill(3)}'
            run = f'{str(run_no).zfill(3)}'
            rec = f'{subject}_{session}_{run}'
            scores_dict[rec] = scores.iloc[i, j]

    filtered_labels = {rec: scores_dict[rec] for rec in valid_recs}
    return filtered_labels

def get_pss_labels(valid_recs, filename='Data/STAI_grading.xlsx', threshold=4):
    """
    Get filtered and binarized PSS (Perceived Stress Scale) scores for a list of valid record IDs.

    Parameters
    ----------
    valid_recs : list of str
        A list of valid record IDs to keep in the filtered scores.
    filename : str, optional
        The path to the Excel file containing the PSS scores. Default is 'Data/STAI_grading.xlsx'.
    threshold : int or float, optional
        The threshold value to use for binarizing the PSS scores. Scores greater than `threshold` will be
        binarized to 1, and scores less than or equal to `threshold` will be binarized to 0. Default is 4.

    Returns
    -------
    filtered_scores : dict
        A dictionary containing the filtered and binarized PSS scores, where the keys are record IDs and the
        values are the corresponding scores.
    """
    scores = load_pss_labels(filename, threshold)
    filtered_scores = filter_pss_labels(scores.iloc[:, 1:], valid_recs)
    return filtered_scores

def get_pss_labels(valid_recs, filename='Data/STAI_grading.xlsx', threshold=4):
    """
    Get filtered and binarized PSS (Perceived Stress Scale) scores for a list of valid record IDs.

    Parameters
    ----------
    valid_recs : list of str
        A list of valid record IDs to keep in the filtered scores.
    filename : str, optional
        The path to the Excel file containing the PSS scores. Default is 'Data/STAI_grading.xlsx'.
    threshold : int or float, optional
        The threshold value to use for binarizing the PSS scores. Scores greater than `threshold` will be
        binarized to 1, and scores less than or equal to `threshold` will be binarized to 0. Default is 4.

    Returns
    -------
    filtered_scores : dict
        A dictionary containing the filtered and binarized PSS scores, where the keys are record IDs and the
        values are the corresponding scores.
    """
    scores = load_pss_labels(filename, threshold)
    filtered_scores = filter_pss_labels(scores.iloc[:, 1:], valid_recs)
    return filtered_scores

def compute_stai_scores(path='Data/STAI_grading.xlsx'):
    """
    Compute the average scores of subjects from the raw scores.
    Parameters
    ----------
    path : str
        Path to the raw scores excel file.
    Returns
    -------
    scores_df : pandas.DataFrame
        DataFrame containing the average scores of subjects
    """

    columns = ['SubjectNo', 'D1Y1', 'D2Y1', 'J1Y1', 'J2Y1']

    try:
        xl = pd.ExcelFile(path)
    except FileNotFoundError:
        logging.error(f"File not found: {path}")
        return

    n_subjects = len(xl.sheet_names)-2
    scores = []

    sheet_names = [name for name in xl.sheet_names if name not in [
        'MAL', 'Rating 1-10']]

    for sheet_name in sheet_names:
        sheet = xl.parse(sheet_name)

        y1_indices = sheet[sheet.columns[0]] == 'Total score Y1'
        y1_scores = sheet.loc[y1_indices].values[0][1::2]

        scores.append(np.concatenate([y1_scores]))
    scores_df = pd.DataFrame(scores, columns=columns[1:])
    scores_df.insert(0, columns[0], range(1, n_subjects+1))
    return scores_df

def compute_stai_labels(scores_df, cutoff):
    """
    Convert scores to labels based on low and high cutoffs.

    Parameters
    ----------
    scores_df : pd.DataFrame
        Dataframe containing the scores for all subjects.
    low_cutoff : int, optional
        The lower cutoff value. Default is 37.

    high_cutoff : int, optional
        The upper cutoff value. Default is 45.

    Returns
    -------
    numpy.ndarray
        An array with shape `(n_subjects, n_sessions * n_runs)` containing
        labels. Each row corresponds to a subject, and each column
        corresponds to a session/run. The values in the array are integers
        representing the label (0, 1, or 2) for that subject, session, and
        run.
    """

    n_subjects = scores_df.shape[0]
    n_sessions = var.NUM_SESSIONS
    n_runs = var.NUM_RUNS

    labels = {}
    for i in range(n_subjects):
        for j in range(n_sessions*n_runs):
            invalid_flag = False
            scores = scores_df.iloc[i, j+1] #need j+1 because it took in subject number as first value
            if scores == 0: 
                invalid_flag = True
            elif scores < cutoff:
                label = 0
            else:
                label = 1

            if not invalid_flag:
                subject = i + 1
                session = j//n_runs + 1 
                run = j % n_runs + 1
                key = f'P{str(subject).zfill(3)}_S{str(session).zfill(3)}_{str(run).zfill(3)}'
                labels[key] = label
            else:
                print('Invalid')
    return labels


