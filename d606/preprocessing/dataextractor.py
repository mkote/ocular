from scipy.io import loadmat
import numpy as np
from numpy import *
from collections import namedtuple
import os
from sklearn import preprocessing

PATH = os.path.abspath(os.path.join(os.path.dirname(__file__),
                                    '../../matfiles/'))  # Path to math files
HERTZ = 250  # Frequency of obtained data
TRIAL_BEGINNING = 700  # 2.8s * 250Hz
TRIAL_LENGTH = 750  # 3s * 250Hz
JUMP = TRIAL_BEGINNING + TRIAL_LENGTH  # The indices to skip


def load_data(files, type):
    """ Load data for training.
        Input: The specific mat file to load, or a range of files to
               load eg. 3 for file 3 or [1, 5] for file 1 to 5
               Type of data: "T" = Training and "E" = Evaluation
        Output: list of 9 * 'number of mat files' Tuples containing:
               25 x m matrix of sensors and measurements
               list of trial start points
               list of class labels for trials
               list of trials with artifacts """

    eeg_list = []  # List of EEG tuples
    start, end = files, files  # Variables for start and end files to load

    if isinstance(files, list):
        start, end = files[0], files[1]

    file_numbers = [int(x) for x in range(start, end + 1)]  # file number load

    for k in file_numbers:
        full_path = PATH + '/A0' + str(k) + type + '.mat'
        trial = namedtuple('EEG', ['matrix', 'trials', 'labels', 'artifacts'])
        mat_data = loadmat(full_path)

        for i in range(0, len(mat_data['data'][0])):
            data_matrix = []  # 25 x m matrix
            trial_list = []  # List of trial start points
            label_list = []  # List of Class labels
            artifact_list = []  # List of artifacts

            eeg_measures = mat_data['data'][0, i][0, 0][0]
            for j in eeg_measures:
                data_matrix.append(j)
            data_matrix = np.array(data_matrix).transpose()

            trial_points = mat_data['data'][0, i][0, 0][1]
            for j in trial_points:
                trial_list.append(j[0])

            class_labels = mat_data['data'][0, i][0, 0][2]
            for j in class_labels:
                label_list.append(j[0])

            artifacts = mat_data['data'][0, i][0, 0][5]
            for j in artifacts:
                artifact_list.append(j[0])

            t = trial(matrix=data_matrix,
                      trials=trial_list,
                      labels=label_list,
                      artifacts=artifact_list)

            eeg_list.append(t)

    return eeg_list


def extract_trials(matrix, trials):
    """
    :param matrix: numpy matrix
    :param trials: list of trial start points
    :return: new matrix only containing motor imagery
             and a new trial start list
    """
    nm = matrix
    trials.append(len(matrix))  # Initialize trials with a end trial
    num_trials = len(trials) - 1

    for i, trial in enumerate(reversed(trials[0:48])):
        nm = delete(nm, s_[trial + JUMP: trials[num_trials - i]], axis=0)
        nm = delete(nm, s_[trial:trial + TRIAL_BEGINNING], axis=0)

    nm = delete(nm, s_[0:trials[0]], axis=0)
    new_trials = [int(x) * 750 for x in range(0, 48)]
    return nm, new_trials
