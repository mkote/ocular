from scipy.io import loadmat
import numpy as np
from numpy import *
from collections import namedtuple
import os

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
    new_matrix = matrix
    trials.append(len(matrix[1]))  # Initialize trials with a end trial
    num_trials = len(trials) - 1

    for i, trial in enumerate(reversed(trials[0:num_trials])):
        new_matrix = delete(new_matrix,
                            s_[trial + JUMP: trials[num_trials - i]],
                            axis=1)

        new_matrix = delete(new_matrix,
                            s_[trial:trial + TRIAL_BEGINNING],
                            axis=1)

    new_matrix = delete(new_matrix, s_[0:trials[0]], axis=1)
    new_trials = [int(x) * TRIAL_LENGTH for x in range(0, num_trials)]
    return new_matrix, new_trials


def trial_splitter(matrix, trials):
    """
    :param matrix: numpy matrix
    :param trials: list of trial start points
    :return: an array of matrix, one matrix per trial
    """
    matrix_list = []
    trials.append(len(matrix[1]))  # Initialize trials with a end trial
    for i in range(0, len(trials) - 1):
        matrix_list.append(np.transpose([matrix[:, trials[i]:trials[i + 1]]]))

    return matrix_list


def d3_matrix_creator(matrix):
    """
    :param matrix: 2d matrix
    :return: 3d matrix with epochs as first index
    """
    slice_list = []
    num_trials = 48
    for x in range(num_trials):
        slice_list.append(matrix[:, x*TRIAL_LENGTH:(x+1)*TRIAL_LENGTH])

    d3_data = array(slice_list)
    return d3_data


def create_events(labels):
    event_list = []
    for label in labels:
        event_list.append([0, 0, label])

    event_data = array(event_list)
    return event_data


def csp_label_reformat(label, type):
    label_list = []
    classes = [1, 2, 3, 4]
    classes.remove(type)
    for lab in label:
        if lab == type:
            label_list.append(int(type))
        else:
            label_list.append(int(''.join([str(x) for x in classes])))

    return label_list


def run_combiner(run_list):
    n_matrix, n_trials, n_labels, n_artifacts = empty([25, 0]), [], [], []
    matrix_length = []
    for i, run in enumerate(run_list):
        # We don't want any runs with only eog data
        if i in [0, 1, 2]:
            continue
        matrix, trials, labels, artifacts = run
        matrix_length.append(matrix.shape[1])
        n_matrix = concatenate((n_matrix, matrix), axis=1)
        n_trials = concatenate((n_trials, trials), axis=1)
        n_labels = concatenate((n_labels, labels), axis=1)
        n_artifacts = concatenate((n_artifacts, artifacts), axis=1)

    n_trials = map(int, trial_time_fixer(n_trials, matrix_length))
    n_matrix = array(n_matrix)
    return n_matrix, n_trials, n_labels, n_artifacts


def trial_time_fixer(trial_list, matrix_length):
    n_trial_list = []
    trial_adder = 0
    matrix_counter = 0
    for i in range(1, len(trial_list)):
        if trial_list[i] > trial_list[i - 1]:
            n_trial_list.append(trial_list[i - 1] + trial_adder)
        else:
            n_trial_list.append(trial_list[i - 1] + trial_adder)
            trial_adder += matrix_length[matrix_counter]
            matrix_counter += 1
    n_trial_list.append(trial_list[-1] + trial_adder)
    return n_trial_list
