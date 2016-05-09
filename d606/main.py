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

    return feature_vector_list


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

        selector = mifs.MutualInformationFeatureSelector(method="JMIM",
                                                         verbose=2,
                                                         categorical=True,
                                                         n_features=4)

        mifs_list = create_mifs_list(selector,
                                     train_features,
                                     len(filt),
                                     n_comp,
                                     train_labels)

        train_features = [mifs_list[i].transform(train_features[i])
                          for i in range(len(mifs_list))]
        test_features = [mifs_list[i].transform(test_features[i])
                         for i in range(len(mifs_list))]


        rfc_list = []
        for i in range(len(train_features)):
            rfc = RandomForestClassifier(n_estimators=n_trees)
            scaled = StandardScaler().fit_transform(train_features[i].tolist())
            rfc.fit(scaled, binarize_labels(train_labels, i))
            rfc_list.append(rfc)


        proba = []
        for i in range(len(train_features)):
            rfc = rfc_list[i]
            scaled = StandardScaler().fit_transform(test_features[i].tolist())
            temp_proba = []
            for j in range(len(scaled)):
                temp_proba.append(rfc.predict_proba(scaled[j].reshape(1, -1)))
            proba.append(temp_proba)

        predictions = []
        for prob in zip(*proba):
            prob = [p[0][0] for p in prob]
            maxprob = max(prob)
            idx = prob.index(maxprob)
            predictions.append(idx + 1)

        accuracy = np.mean([a == b for (a, b) in zip(predictions, test_labels)])
        print("Accuracy: " + str(accuracy * 100) + "%")

        accuracies.append(accuracy)

    os.chdir(old_path)
    return np.mean(accuracies) * 100, time.time()
