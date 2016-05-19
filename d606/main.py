import time
import os
import sys
import filehandler
import numpy as np
from skll import metrics
from sklearn.ensemble import RandomForestClassifier
from eval.timing import timed_block
from featureselection.mnecsp import csp_one_vs_all
from preprocessing.dataextractor import load_data, restructure_data, separate_eog_eeg, d3_matrix_creator
from preprocessing.filter import Filter
from preprocessing.trial_remaker import remake_trial
from itertools import chain
from multiprocessing import freeze_support
from numpy import array
from resultparser import best_subject_params
from preprocessing.oaclbase import OACL

RUNS_WITH_EEG = array(range(-6, 0))


def main(n_comp, band_list, subject, oacl_ranges=None, m=None, thetas=None):
    print 'Running with following args \n'
    print n_comp, band_list, subject, oacl_ranges, m, thetas

    old_path = os.getcwd()
    os.chdir('..')

    train = load_data(subject, "T")
    _, train = separate_eog_eeg(train)

    oacl = OACL(ranges=oacl_ranges, m=m, multi_run=True, trials=False)
    oacl.theta = thetas

    train, _ = remake_trial(train, m=m, oacl_ranges=oacl_ranges, arg_oacl=oacl)

    test_indexes = [[i] for i in range(6)]
    train_indexes = [range(i) + range(i+1, 6) for i in range(6)]

    kappas = []
    accuracies = []
    for train_index, test_index in zip(train_indexes, test_indexes):
        _train, _test = transform_fold_data(train_index, test_index, train)
        accuracy, kappa = evaluate_fold(_train, _test, band_list, n_comp)
        print("Accuracy: " + str(accuracy * 100) + "%")
        print("Kappa: " + str(kappa))
        kappas.append(kappa)
        accuracies.append(accuracy)

    os.chdir(old_path)
    print "Mean accuracy: " + str(np.mean(accuracies) * 100)
    print "Mean kappa: " + str(np.mean(kappas))
    return np.mean(accuracies) * 100, time.time()


def run_oacl(subject, runs, oacl_ranges, m):
    # Generate a name for serializing of file
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


def transform_fold_data(train_index, test_index, train):
    new_train = array(train)[RUNS_WITH_EEG[train_index]]
    new_test = array(train)[RUNS_WITH_EEG[test_index]]

    return new_train, new_test


def evaluate_fold(train, test, band_list, n_comp, seeds=(4, 8, 15, 16, 23, 42)):
    filters = Filter(band_list)

    train_bands, train_labels = restructure_data(train, filters)
    test_bands, test_labels = restructure_data(test, filters)

    csp_list = []
    for band in train_bands:
        csp_list.append(csp_one_vs_all(band, 4, n_comps=n_comp))

    train_features = create_feature_vector_list(train_bands, csp_list)
    test_features = create_feature_vector_list(test_bands, csp_list)

    kappas = []
    accuracies = []
    for seed in seeds:
        rf = RandomForestClassifier(n_estimators=len(band_list) * 4 * n_comp, random_state=seed)
        rf.fit(train_features, train_labels)

        predictions = []
        for y in test_features:
            predictions.append(rf.predict(array(y).reshape(1, -1)))

        kappa = metrics.kappa(test_labels, predictions)
        accuracy = np.mean([a == b for (a, b) in zip(predictions, test_labels)])

        kappas.append(kappa)
        accuracies.append(accuracy)

    return np.mean(accuracies), np.mean(kappas)


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


def evaluate(n_comp, band_list, subject, oacl_ranges=None, m=None, thetas=None):
    old_path = os.getcwd()
    os.chdir('..')

    train = load_data(subject, "T")
    _, train = separate_eog_eeg(train)
    test = load_data(subject, "E")
    _, test = separate_eog_eeg(test)

    oacl = OACL(ranges=oacl_ranges, m=m, multi_run=True, trials=False)
    oacl.theta = thetas

    train, _ = remake_trial(train, m=m, oacl_ranges=oacl_ranges, arg_oacl=oacl)
    test, _ = remake_trial(test, m=m, oacl_ranges=oacl_ranges, arg_oacl=oacl)

    train = array(train)[RUNS_WITH_EEG]
    test = array(test)[RUNS_WITH_EEG]
    accuracy, kappa = evaluate_fold(train, test, band_list, n_comp)
    
    print("Accuracy: " + str(accuracy * 100) + "%")
    print("Kappa: " + str(kappa))

    os.chdir(old_path)


def translate_params(par):
    n_comp = int(par[0])
    band_range = int(par[1])
    num_bands = int(36 / band_range)
    band_list = [[4 + band_range * x, 4 + band_range * (x + 1)] for x in range(num_bands)]
    if len(par) > 2:
        s = int(par[2])
        r = int(par[3])
        m = int(par[4]) * 2 + 1
        oacl_range = ((s, s + r), )
        thvals = [array([float(par[x])]) for x in xrange(5, 27)]
    else:
        m = None
        oacl_range = None
        thvals = None

    return n_comp, band_list, oacl_range, m, thvals


# Input: subject, n_comp, band_range[, s, r1, r2, space, m]
if __name__ == '__main__':
    freeze_support()
    errors = []
    if len(sys.argv) > 1:
        subject = int(sys.argv[1])
        error, params = best_subject_params(subject)
        n_comp, band_list, oacl_range, m, thvals = translate_params(params[0][2:])
        evaluate(n_comp, band_list, subject, oacl_range, m, thvals)
    else:
        print("No arguments passed - continuing with default parameters.")
        main(12, [[4, 9], [9, 14], [14, 19], [19, 24], [24, 29], [29, 34], [34, 39]], 1, ((2, 3),), 7, [array([0.1]) for x in range(22)])
