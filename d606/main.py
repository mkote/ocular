from collections import namedtuple
import time
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import preprocessing.searchgrid as search
from eval.timing import timed_block
from featureselection import mifs
from featureselection.mifsaux import create_mifs_list, binarize_labels
from featureselection.mnecsp import csp_one_vs_all
from preprocessing.dataextractor import load_data, restructure_data, separate_eog_eeg, d3_matrix_creator
from preprocessing.filter import Filter
from preprocessing.trial_remaker import remake_trial, remake_single_run_transform
from itertools import chain
from sklearn import cross_validation
from multiprocessing import freeze_support
from numpy import array
from preprocessing.oaclbase import OACL
import os
import filehandler
import numpy as np

optimize_params = True


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


def main(*args):
    print 'Running with following args \n'
    print args
    named_grid = namedtuple('Grid', ['n_comp', 'n_trees', 'band_list', 'oacl_ranges', 'm', 'subject'])
    search.grid = named_grid(*args)

    old_path = os.getcwd()
    os.chdir('..')

    # Load args from search-grid
    oacl_ranges = search.grid.oacl_ranges if 'oacl_ranges' in search.grid._fields else ((3, 7), (7, 15))
    m = search.grid.m if 'm' in search.grid._fields else 11
    n_trees = search.grid.n_trees if 'n_trees' in search.grid._fields else 20
    filt = search.grid.band_list if 'band_list' in search.grid._fields else [[8, 12], [16, 24]]
    n_comp = search.grid.n_comp if 'n_comp' in search.grid._fields else 3
    subject = search.grid.subject if 'subject' in search.grid._fields else 1

    # Generate a name for serializing of file
    filename_suffix = filehandler.generate_filename(oacl_ranges, m, subject)

    # Check whether the data is already present as serialized data
    # If not run OACL and serialize, else load data from file
    if filehandler.file_is_present('runs' + filename_suffix) is False:
        with timed_block('Iteration '):
            runs = load_data(subject, "T")
            eog_test, runs = separate_eog_eeg(runs)
            runs, train_oacl = remake_trial(runs)

            thetas = train_oacl.trial_thetas

        # Save data, could be a method instead
        filehandler.save_data(runs, 'runs' + filename_suffix)
        filehandler.save_data(thetas, 'thetas' + filename_suffix)
    else:
        runs = filehandler.load_data('runs' + filename_suffix)
        thetas = filehandler.load_data('thetas' + filename_suffix)

    run_choice = range(3, 9)

    sh = cross_validation.ShuffleSplit(6, n_iter=6, test_size=0.16)

    accuracies = []
    for train_index, test_index in sh:
        train = array(runs)[array(run_choice)[(sorted(train_index))]]
        test = load_data(subject, "T")
        _, test = separate_eog_eeg(test)
        test = array(test)[array(run_choice)[test_index]]

        oacl = OACL(ranges=oacl_ranges, m=m, multi_run=True)
        oacl.theta = oacl.generalize_thetas(array(thetas)[train_index])

        test = remake_single_run_transform(test, oacl)

        filters = Filter(filt)


        train_bands, train_labels = restructure_data(train, filters)
        test_bands, test_labels = restructure_data(test, filters)

        csp_list = []
        for band in train_bands:
            csp_list.append(csp_one_vs_all(band, 4, n_comps=n_comp))

        train_features = create_feature_vector_list(train_bands, csp_list)
        test_features = create_feature_vector_list(test_bands, csp_list)


        rf = RandomForestClassifier(n_estimators=n_trees)
        rf.fit(train_features, train_labels)
        important_features = rf.feature_importances_
        indices = np.argsort(important_features)[::-1]

        bahm_magic = int((n_comp/2)*np.log2(2*len(filt)))
        indices = indices[0:bahm_magic]

        temp_train = [array(x)[indices] for x in train_features]
        rf = RandomForestClassifier(n_estimators=bahm_magic)
        rf.fit(temp_train, train_labels)

        temp_test = [array(x)[indices] for x in test_features]
        predictions = []
        for y in temp_test:
            predictions.append(rf.predict(y.reshape(1, -1)))

        accuracy = np.mean([a == b for (a, b) in zip(predictions, test_labels)])
        print("Accuracy: " + str(accuracy * 100) + "%")

        accuracies.append(accuracy)

    os.chdir(old_path)
    print "Mean accuracy: " + str(np.mean(accuracies) * 100)
    return np.mean(accuracies) * 100, time.time()

if __name__ == '__main__':
    freeze_support()
    main(12, 29, [[4, 9], [9, 14], [14, 19], [19, 24], [24, 29], [29, 34], [34, 39]], ((2, 3), (4, 5)), 7, 1)

