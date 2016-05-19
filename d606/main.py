from __future__ import absolute_import
import time
import os
import sys

from IPython.core import magic

import filehandler
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from eval.timing import timed_block
from featureselection.mnecsp import csp_one_vs_all
from preprocessing.dataextractor import load_data, restructure_data, separate_eog_eeg, d3_matrix_creator, extract_trials_two
from preprocessing.filter import Filter
from preprocessing.trial_remaker import remake_trial, remake_single_run_transform
from itertools import chain
from multiprocessing import freeze_support
from numpy import array
from preprocessing.oaclbase import OACL
from preprocessing.oacl import find_artifact_signals, extract_trials_array
from autograd import grad
from autograd.util import quick_grad_check
import autograd.numpy as auto

from real_main import magic

RUNS_WITH_EEG = array(range(-6, 0))
inputs = 0
raw_signal = 0
targets = 0


def main(n_comp, band_list, subject, oacl_ranges=None, m=None):
    print 'Running with following args \n'
    print n_comp, band_list, subject, oacl_ranges, m

    old_path = os.getcwd()
    os.chdir('..')

    train = load_data(subject, "T")
    _, train = separate_eog_eeg(train)
    test = load_data(subject, "T")
    _, test = separate_eog_eeg(test)
    thetas = None

    if not any([x == None for x in [oacl_ranges, m]]):
        thetas, train = run_oacl(subject, train, oacl_ranges, m)

    test_indexes = [[i] for i in range(6)]
    train_indexes = [range(i) + range(i+1, 6) for i in range(6)]

    accuracies = []
    for train_index, test_index in zip(train_indexes, test_indexes):
        _train, _test = transform_fold_data(train_index, test_index, train, test,
                            oacl_ranges, thetas, m)
        accuracy = evaluate_fold(_train, _test, band_list, n_comp)
        print("Accuracy: " + str(accuracy * 100) + "%")
        accuracies.append(accuracy)

    os.chdir(old_path)
    print "Mean accuracy: " + str(np.mean(accuracies) * 100)
    return np.mean(accuracies) * 100, time.time()


def rearrange_target(target, label):
    new_targets = []
    for tar in target:
        if tar == label:
            new_targets.append(True)
        else:
            new_targets.append(False)
    return np.array(new_targets)


def generate_thetas(runs, ranges, m):
    global inputs, raw_signal, targets
    raw_signals = np.array([], dtype=float).reshape(22, 0)
    labels = []
    artifact_list = []
    start_trial_times = runs[3][1]
    new_start_times = []
    weight_list = []
    with timed_block("Extract Trials!"):
        for i, run in enumerate(runs[3:]):
            tri = run[0]
            labels.extend(run[2])
            raw_signals = np.concatenate((raw_signals, np.array(tri)), axis=1)

        appender = len(runs[3][0][0])
        for i in xrange(6):
            new_start_times.extend([x + (appender * i) for x in start_trial_times])

        for i, channel in enumerate(raw_signals):
            raw_signal = []
            raw_signal = extract_trials_array(channel, new_start_times)
            #for raw in temp_raw_signal:
            #    raw_signal.extend(raw)
            inputs = raw_signal

            magic(inputs, labels, m, ranges)


def run_oacl(subject, runs, oacl_ranges, m):
    # Generate a name for serializing of file
    generate_thetas(runs, oacl_ranges, m)
    filename_suffix = filehandler.generate_filename(oacl_ranges, m, subject)

    # Check whether the data is already present as serialized data
    # If not run OACL and serialize, else load data from file
    if filehandler.file_is_present('runs' + filename_suffix) is False:
        with timed_block('Iteration '):
            runs, train_oacl = remake_trial(runs, m=m, oacl_ranges=oacl_ranges)
            thetas = train_oacl.trial_thetas

        # Save data, could be a method instead
        filehandler.save_data(runs, 'runs' + filename_suffix)
        filehandler.save_data(thetas, 'thetas' + filename_suffix)
    else:
        runs = filehandler.load_data('runs' + filename_suffix)
        thetas = filehandler.load_data('thetas' + filename_suffix)

    return thetas, runs


def transform_fold_data(train_index, test_index, train, test,
                        oacl_ranges=None, thetas=None, m=None):
    train = array(train)[RUNS_WITH_EEG[train_index]]
    test = array(test)[RUNS_WITH_EEG[test_index]]

    if not any([x == None for x in [oacl_ranges, thetas, m]]):
        oacl = OACL(ranges=oacl_ranges, m=m, multi_run=True)
        oacl.theta = oacl.generalize_thetas(array(thetas)[train_index])
        test = remake_single_run_transform(test, oacl)

    return train, test


def evaluate_fold(train, test, band_list, n_comp, seeds=(4, 8, 15, 16, 23, 42)):
    filters = Filter(band_list)

    train_bands, train_labels = restructure_data(train, filters)
    test_bands, test_labels = restructure_data(test, filters)

    csp_list = []
    for band in train_bands:
        csp_list.append(csp_one_vs_all(band, 4, n_comps=n_comp))

    train_features = create_feature_vector_list(train_bands, csp_list)
    test_features = create_feature_vector_list(test_bands, csp_list)

    accuracies = []
    for seed in seeds:
        rf = RandomForestClassifier(n_estimators=len(band_list) * 4 * n_comp, random_state=seed)
        rf.fit(train_features, train_labels)

        predictions = []
        for y in test_features:
            predictions.append(rf.predict(array(y).reshape(1, -1)))

        accuracy = np.mean([a == b for (a, b) in zip(predictions, test_labels)])
        accuracies.append(accuracy)

    return np.mean(accuracies)


def create_feature_vector_list(bands, csp_list):
    feature_list = []
    temp = []
    for band, csp in zip(bands, csp_list):
        d3_matrix = d3_matrix_creator(band[0], len(band[1]))
        for single_csp in csp:
            temp.append(single_csp.transform(d3_matrix))
        feature_list.append(temp)
        temp = []

    feature_vector_list = []
    for x in zip(*feature_list):
        feature_vector_list.append(array([list(chain(*z)) for z in zip(*x)]))

    combo = [list(chain(*z)) for z in zip(*feature_vector_list)]

    return combo


def evaluate(n_comp, band_list, subject, oacl_ranges=None, m=None):
    old_path = os.getcwd()
    os.chdir('..')

    train = load_data(subject, "T")
    _, train = separate_eog_eeg(train)
    test = load_data(subject, "E")
    _, test = separate_eog_eeg(test)
    if not any([x == None for x in [oacl_ranges, m]]):
        thetas, train = run_oacl(subject, train, oacl_ranges, m)
        oacl = OACL(ranges=oacl_ranges, m=m, multi_run=True)
        oacl.theta = oacl.generalize_thetas(array(thetas))
        test, _ = remake_trial(test, arg_oacl=oacl)

    train = array(train)[RUNS_WITH_EEG]
    test = array(test)[RUNS_WITH_EEG]
    accuracy = evaluate_fold(train, test, band_list, n_comp)
    print("Accuracy: " + str(accuracy * 100) + "%")

    os.chdir(old_path)


def translate_params(par):
    n_comp = int(par[0])
    band_range = int(par[1])
    num_bands = int(36 / band_range)
    band_list = [[4 + band_range * x, 4 + band_range * (x + 1)] for x in range(num_bands)]
    if len(par) > 2:
        s = int(par[2])
        r1 = int(par[3])
        r2 = int(par[4])
        space = int(par[5])
        m = int(par[6]) * 2 + 1
        oacl_ranges = ((s, s + r1), (space + s + r1, space + s + r1 + r2))
    else:
        m = None
        oacl_ranges = None

    return n_comp, band_list, oacl_ranges, m


# Input: subject, n_comp, band_range[, s, r1, r2, space, m]
if __name__ == '__main__':
    freeze_support()
    errors = []
    if len(sys.argv) > 1:
        n_comp, band_list, oacl_ranges, m = translate_params(sys.argv[2:])
        subject = int(sys.argv[1])
        evaluate(n_comp, band_list, subject, oacl_ranges, m)
    else:
        print("No arguments passed - continuing with default parameters.")
        evaluate(4, [[4, 8], [8, 12], [12, 16], [16, 20], [20, 24], [24, 28], [28, 32], [32, 36], [36, 40]], 5, ((2, 3), (3 , 30)), 7)
