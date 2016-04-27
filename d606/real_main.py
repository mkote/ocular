import cPickle
from collections import namedtuple
import preprocessing.searchgrid as search
from classification.svm import csv_one_versus_all, svm_prediction
from eval.score import scoring
from eval.timing import timed_block
from eval.voting import csp_voting
from featureselection.mnecsp import csp_one_vs_all
from preprocessing.dataextractor import load_data, restructure_data, extract_eog, d3_matrix_creator
from classification.randomforrest import rfl_one_versus_all, rfl_prediction
from preprocessing.filter import Filter
from preprocessing.trial_remaker import remake_trial, remake_single_run_transform
from os import listdir
from itertools import chain
from os.path import isfile, join
from multiprocessing import freeze_support
from sklearn import cross_validation
from numpy import array
from skfeature.function.information_theoretical_based.MIFS import mifs
from preprocessing.oaclbase import OACL
import os

optimize_params = True


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

    for train_index, test_index in sh:
        csp_list = []

        train = array(runs)[array(run_choice)[(sorted(train_index))]]
        test = load_data(8, "T")
        eog_test, test = extract_eog(test)
        test = array(test)[array(run_choice)[test_index]]

        m = search.grid.m if 'm' in search.grid._fields else 11
        ranges = search.grid.oacl_ranges if 'oacl_ranges' in search.grid._fields else ((3, 7), (7, 15))
        oacl = OACL(ranges=ranges, m=m, multi_run=True)
        oacl.theta = oacl.generalize_thetas(array(thetas)[train_index])
        test = remake_single_run_transform(test, oacl)

        filt = search.grid.band_list if 'band_list' in search.grid._fields else [[8, 12], [16, 24]]
        filters = Filter(filt)

        n_comp = search.grid.n_comp if 'n_comp' in search.grid._fields else 3

        train_bands, train_combined_labels = restructure_data(train, filters)
        test_bands, test_combined_labels = restructure_data(test, filters)

        for band in train_bands:
            csp_list.append(csp_one_vs_all(band, 4, n_comps=n_comp))

        feature_list = []
        temp = []
        for band, csp in zip(test_bands, csp_list):
            d3_matrix = d3_matrix_creator(band[0], len(band[1]))
            for single_csp in csp:
                temp.append(single_csp.transform(d3_matrix))
            feature_list.append(temp)
            temp = []

        f = []
        for i, x in enumerate(zip(*list(chain(*feature_list)))):
            y = array([float(test_combined_labels[i]) for z in range(len(x))])
            t = []
            t.append(array(x))
            f.append(mifs(array(t), test_combined_labels[i], kwargs={'n_selected_features': 4}))


        # Use MIFS

        print "Done so far"


    os.chdir(old_path)

    with timed_block('All Time'):
        # for subject in [int(x) for x in range(1, 2)]:
        csp_list = []
        svc_list = []
        rfl_list = []
        filt = search.grid.band_list if 'band_list' in search.grid._fields else [[8, 12], [16, 24]]
        filters = Filter(filt)

        train_bands, train_combined_labels = restructure_data(runs, filters)

        test_bands, test_combined_labels = restructure_data(evals, filters)

        # CSP one VS all, give csp_one_cs_all num_different labels as input


        # Create a svm for each csp and band
        for csp, band in zip(csp_list, train_bands):
            svc_list.append(csv_one_versus_all(csp, band))

        # Create a random forest tree for each csp and band
        for csp, band in zip(csp_list, train_bands):
            rfl_list.append(rfl_one_versus_all(csp, band))

        # Predict results with svm's
        svm_results = svm_prediction(test_bands, svc_list, csp_list)

        # Predict results with svm's
        rfl_results = rfl_prediction(test_bands, rfl_list, csp_list)

        svm_voting_results = csp_voting(svm_results)
        rfl_voting_results = csp_voting(rfl_results)

        svm_score, wrong_list = scoring(svm_voting_results, test_combined_labels)
        rfl_score, wrong_list = scoring(rfl_voting_results, test_combined_labels)
        score = svm_score if svm_score >= rfl_score else rfl_score
    print search.grid
    print 'rfl results: ' + str(rfl_score)
    print 'svm results ' + str(svm_score)
    print '\n'
    print 'Best score: ' + str(score)
    return svm_score, 1200

if __name__ == '__main__':
    freeze_support()
    main(2, 0.1, 'rbf', [[4, 8], [8, 12], [12, 16], [16, 20], [20, 30]], ((3, 7), (7, 15)), 11)
