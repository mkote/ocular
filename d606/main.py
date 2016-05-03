import cPickle
from collections import namedtuple

from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

import preprocessing.searchgrid as search
from eval.timing import timed_block
from featureselection import mifs
from featureselection.mifsaux import create_mifs_list, binarize_labels
from featureselection.mnecsp import csp_one_vs_all
from preprocessing.dataextractor import load_data, restructure_data, extract_eog, d3_matrix_creator
from preprocessing.filter import Filter
from preprocessing.trial_remaker import remake_trial, remake_single_run_transform
from os import listdir
from itertools import chain
from os.path import isfile, join
from multiprocessing import freeze_support
from sklearn import cross_validation
from numpy import array
from preprocessing.oaclbase import OACL
import os
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
    named_grid = namedtuple('Grid', ['n_comp', 'C', 'kernel', 'band_list', 'oacl_ranges', 'm'])
    search.grid = named_grid(*args)
    runs, evals, thetas = '', '', ''

    old_path = os.getcwd()
    os.chdir('..')

    oacl_ranges = search.grid.oacl_ranges
    pickel_file_name = str(oacl_ranges[0][0]) + str(oacl_ranges[0][1]) + str(oacl_ranges[1][0])
    pickel_file_name += str(oacl_ranges[1][1]) + str(search.grid.m) + '.dump'
    if not os.path.isdir('pickelfiles'):
        os.mkdir('pickelfiles')
    onlyfiles = [f for f in listdir('pickelfiles') if isfile(join('pickelfiles', f))]
    if len(onlyfiles) >= 100:
        file_to_delete1 = onlyfiles[0].replace('evals', '')
        file_to_delete1 = file_to_delete1.replace('runs', '')
        file_to_delete2 = 'runs' + file_to_delete1
        file_to_delete1 = 'evals' + file_to_delete1
        os.remove('pickelfiles/' + file_to_delete1)
        os.remove('pickelfiles/' + file_to_delete2)
        onlyfiles.remove(file_to_delete1)
        onlyfiles.remove(file_to_delete2)
    if 'runs' + pickel_file_name not in onlyfiles:
        with timed_block('Iteration '):
            runs = load_data(8, "T")
            eog_test, runs = extract_eog(runs)
            runs, train_oacl = remake_trial(runs)

            thetas = train_oacl.trial_thetas

            evals = load_data(8, "E")
            eog_eval, evals = extract_eog(evals)
            evals, test_oacl = remake_trial(evals, arg_oacl=train_oacl)

        # Save data, could be a method instead
        with open('pickelfiles/runs' + pickel_file_name, "wb") as output:
            cPickle.dump(runs, output, cPickle.HIGHEST_PROTOCOL)

        with open('pickelfiles/evals' + pickel_file_name, "wb") as output:
            cPickle.dump(evals, output, cPickle.HIGHEST_PROTOCOL)

        with open('pickelfiles/thetas' + pickel_file_name, 'wb') as output:
            cPickle.dump(thetas, output, cPickle.HIGHEST_PROTOCOL)
    else:
        with open('pickelfiles/runs' + pickel_file_name, "rb") as input:
            runs = cPickle.load(input)

        with open('pickelfiles/evals' + pickel_file_name, "rb") as input:
            evals = cPickle.load(input)

        with open('pickelfiles/thetas' + pickel_file_name, 'rb') as input:
            thetas = cPickle.load(input)

    run_choice = range(3, 9)

    sh = cross_validation.ShuffleSplit(6, n_iter=6, test_size=0.16)

    accuracies = []
    for train_index, test_index in sh:
        train = array(runs)[array(run_choice)[(sorted(train_index))]]
        test = load_data(8, "T")
        eog_test, test = extract_eog(test)
        test = array(test)[array(run_choice)[test_index]]

        C = search.grid.C if 'C' in search.grid._fields else 1
        kernel = search.grid.kernel if 'kernel' in search.grid._fields else 'linear'
        m = search.grid.m if 'm' in search.grid._fields else 11
        ranges = search.grid.oacl_ranges if 'oacl_ranges' in search.grid._fields else ((3, 7), (7, 15))
        oacl = OACL(ranges=ranges, m=m, multi_run=True)
        oacl.theta = oacl.generalize_thetas(array(thetas)[train_index])

        test = remake_single_run_transform(test, oacl)

        filt = search.grid.band_list if 'band_list' in search.grid._fields else [[8, 12], [16, 24]]
        filters = Filter(filt)

        n_comp = search.grid.n_comp if 'n_comp' in search.grid._fields else 3

        train_bands, train_labels = restructure_data(train, filters)
        test_bands, test_labels = restructure_data(test, filters)

        csp_list = []
        for band in train_bands:
            csp_list.append(csp_one_vs_all(band, 4, n_comps=n_comp))

        train_features = create_feature_vector_list(train_bands, csp_list)
        test_features = create_feature_vector_list(test_bands, csp_list)

        selector = mifs.MutualInformationFeatureSelector(method="JMI",
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


        svc_list = []
        for i in range(len(train_features)):
            svc = SVC(C=C, kernel=kernel, gamma='auto', probability=True)
            scaled = StandardScaler().fit_transform(train_features[i].tolist())
            svc.fit(scaled, binarize_labels(train_labels, i))
            svc_list.append(svc)


        proba = []
        for i in range(len(train_features)):
            svc = svc_list[i]
            scaled = StandardScaler().fit_transform(test_features[i].tolist())
            temp_proba = []
            for j in range(len(scaled)):
                temp_proba.append(svc.predict_proba(scaled[j]))
            proba.append(temp_proba)

        predictions = []
        for prob in zip(*proba):
            prob = [p[0][0] for p in prob]
            maxprob = max(prob)
            idx = prob.index(maxprob)
            predictions.append(idx + 1)

        accuracy = np.mean([1 if a == b else 0 for (a, b) in zip(predictions, test_labels)])
        print("Accuracy: " + str(accuracy) + "%")

        accuracies.append(accuracy)

    return np.mean(accuracies) * 100



if __name__ == '__main__':
    freeze_support()
    main(2, 0.1, 'rbf', [[4, 8], [8, 12], [12, 16], [16, 20], [20, 30]], ((3, 7), (7, 15)), 11)
