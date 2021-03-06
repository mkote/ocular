from scipy.io import loadmat
import numpy as np
from numpy import *
from collections import namedtuple
import os
from itertools import chain

PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../matfiles/'))  # Path to math files
HERTZ = 250  # Frequency of obtained data
TRIAL_BEGINNING = 700  # 2.8s * 250Hz
TRIAL_LENGTH = 750  # 3s * 250Hz
JUMP = TRIAL_BEGINNING + TRIAL_LENGTH  # The indices to skip
EEG = namedtuple('EEG', ['matrix', 'trials', 'labels', 'artifacts'])


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

            # t = trial(matrix=data_matrix,
            #           trials=trial_list,
            #           labels=label_list,
            #           artifacts=artifact_list)

            eeg_list.append((data_matrix,trial_list, label_list, artifact_list))

    return eeg_list


def extract_trials_single_channel(matrix, trials):
    new_matrix = []
    num_trials = len(trials)
    for trial in trials:
        new_matrix.extend(transpose(matrix[trial+TRIAL_BEGINNING:trial+JUMP]))
    return transpose(new_matrix), [int(x) * TRIAL_LENGTH for x in range(0, num_trials)]


def extract_trials_two(matrix, trials):
    new_matrix = []
    num_trials = len(trials)
    for trial in trials:
        new_matrix.extend(transpose(matrix[0:len(matrix),
                                    trial+TRIAL_BEGINNING:trial+JUMP]))
    return transpose(new_matrix), [int(x) * TRIAL_LENGTH for x in range(0, num_trials)]


def d3_matrix_creator(matrix, num_trials):
    """
    :param matrix: 2d matrix
    :return: 3d matrix with epochs as first index
    """
    slice_list = []
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
    """
    Combine runs
    :param run_list: a list of runs to combine
    :return: a new tuple containing a matrix, trials, labels and
             artifacts for the combined runs
    """
    first_index = (run_list.index(x) for x in run_list if len(x[1]) > 0).next()
    m_matrix = transpose(list(chain(*[transpose(x[0]) for x in run_list[first_index:]])))
    m_trials = list(chain(*[x[1] for x in run_list[first_index:]]))
    m_labels = list(chain(*[x[2] for x in run_list[first_index:]]))
    m_artifacts = list(chain(*[x[3] for x in run_list[first_index:]]))
    matrix_length = [x[0].shape[1] for x in run_list[first_index:]]

    m_trials = map(int, trial_time_fixer(m_trials, matrix_length))
    return m_matrix, m_trials, m_labels, m_artifacts


def trial_time_fixer(trial_list, matrix_length):
    """
    Used in Run combiner, to create a correct list of trial start points
    :param trial_list: A list of Trial start times
    :param matrix_length: list of matrix length for each run
    :return: a new trial list
    """
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


def restructure_data(runs, filters):
    bands = []
    combined_data = []
    combined_labels = []
    filter_bank = []
    num_banks = len(filters.band_range)

    for run in runs:
        matrix, trials, labels, artifacts = run
        filter_bank.append([(single_filter, trials, labels, artifacts) for single_filter in filters.filter(matrix)])

    data_tuple_bands = [[] for x in range(0, len(filter_bank[0]))]
    #  Restructure matrices, and recreate data tuples
    for bank in filter_bank:
        for x in range(0, num_banks):
            data_tuple_bands[x].append(bank[x])
    del bank, filter_bank, runs

    # Call run_combiner with band from data_tuples
    for x in [0 for y in range(0, len(data_tuple_bands))]:
        combined_data.append(run_combiner(data_tuple_bands[x]))
        del data_tuple_bands[x]

    # Trial Extraction before csp and svn
    for eeg_signal in combined_data:
        old_matrix, old_trials, labels, artifacts = eeg_signal
        new_matrix, new_trials = extract_trials_two(old_matrix[0:22], old_trials)
        bands.append((new_matrix, new_trials, labels))
    combined_labels.extend(combined_data[0][2])

    return bands, combined_labels


def separate_eog_eeg(runs):
    n_runs = []
    n_eog = []
    for run in runs:
        matrix, trials, labels, artifacts = run
        eog = matrix[22:25]
        eeg = matrix[0:22]
        n_runs.append((eeg, trials, labels, artifacts))
        n_eog.append(eog)

    return n_eog, n_runs
