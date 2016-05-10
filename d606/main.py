import time
import os
import filehandler
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from eval.timing import timed_block
from featureselection.mnecsp import csp_one_vs_all
from preprocessing.dataextractor import load_data, restructure_data, separate_eog_eeg, d3_matrix_creator
from preprocessing.filter import Filter
from preprocessing.trial_remaker import remake_trial, remake_single_run_transform
from itertools import chain
from sklearn import cross_validation
from multiprocessing import freeze_support
from numpy import array
from preprocessing.oaclbase import OACL

RUNS_WITH_EEG = array(range(3, 9))


def main(n_comp, band_list, subject, oacl_ranges=None, m=None):
    print 'Running with following args \n'
    print n_comp, band_list, subject, oacl_ranges, m

    old_path = os.getcwd()
    os.chdir('..')

    runs = load_data(subject, "T")
    eog_test, runs = separate_eog_eeg(runs)
    thetas = None

    if any([x == None for x in [oacl_ranges, m]]):
        thetas, runs = run_oacl(subject, runs, oacl_ranges, m)

    sh = cross_validation.ShuffleSplit(6, n_iter=6, test_size=0.16)

    accuracies = []
    for train_index, test_index in sh:
        accuracy = evaluate_fold(train_index, test_index, runs, subject, band_list, n_comp,
                                 oacl_ranges, thetas, m)
        print("Accuracy: " + str(accuracy * 100) + "%")
        accuracies.append(accuracy)

    os.chdir(old_path)
    print "Mean accuracy: " + str(np.mean(accuracies) * 100)
    return np.mean(accuracies) * 100, time.time()


def run_oacl(subject, runs, oacl_ranges, m):
    # Generate a name for serializing of file
    filename_suffix = filehandler.generate_filename(oacl_ranges, m, subject)

    # Check whether the data is already present as serialized data
    # If not run OACL and serialize, else load data from file
    if filehandler.file_is_present('runs' + filename_suffix) is False:
        with timed_block('Iteration '):
            runs, train_oacl = remake_trial(runs)

            thetas = train_oacl.trial_thetas

        # Save data, could be a method instead
        filehandler.save_data(runs, 'runs' + filename_suffix)
        filehandler.save_data(thetas, 'thetas' + filename_suffix)
    else:
        runs = filehandler.load_data('runs' + filename_suffix)
        thetas = filehandler.load_data('thetas' + filename_suffix)

    return thetas, runs


def evaluate_fold(train_index, test_index, runs, subject, band_list, n_comp,
                  oacl_ranges=None, thetas=None, m=None):
    train = array(runs)[RUNS_WITH_EEG[sorted(train_index)]]
    test = load_data(subject, "T")
    _, test = separate_eog_eeg(test)
    test = array(test)[RUNS_WITH_EEG[test_index]]

    if not any([x == None for x in[oacl_ranges, thetas, m]]):
        oacl = OACL(ranges=oacl_ranges, m=m, multi_run=True)
        oacl.theta = oacl.generalize_thetas(array(thetas)[train_index])
        test = remake_single_run_transform(test, oacl)

    filters = Filter(band_list)

    train_bands, train_labels = restructure_data(train, filters)
    test_bands, test_labels = restructure_data(test, filters)

    csp_list = []
    for band in train_bands:
        csp_list.append(csp_one_vs_all(band, 4, n_comps=n_comp))

    train_features = create_feature_vector_list(train_bands, csp_list)
    test_features = create_feature_vector_list(test_bands, csp_list)

    rf = RandomForestClassifier(n_estimators=len(band_list) * 4 * n_comp)
    rf.fit(train_features, train_labels)

    predictions = []
    for y in test_features:
        predictions.append(rf.predict(array(y).reshape(1, -1)))

    accuracy = np.mean([a == b for (a, b) in zip(predictions, test_labels)])
    return accuracy


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

    comb = [list(chain(*z)) for z in zip(*feature_vector_list)]

    return comb


if __name__ == '__main__':
    freeze_support()
    main(12, [[4, 9], [9, 14], [14, 19], [19, 24], [24, 29], [29, 34], [34, 39]], 1, ((2, 3), (4, 5)), 7)
