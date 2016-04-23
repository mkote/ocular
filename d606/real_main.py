import cPickle
from collections import namedtuple
import preprocessing.searchgrid as search
from classification.svm import csv_one_versus_all, svm_prediction
from eval.score import scoring
from eval.timing import timed_block
from eval.voting import csp_voting
from featureselection.mnecsp import csp_one_vs_all
from preprocessing.dataextractor import load_data, restructure_data, extract_eog
from classification.randomforrest import rfl_one_versus_all, rfl_prediction
from preprocessing.filter import Filter
from preprocessing.trial_remaker import remake_trial
from os import listdir
from os.path import isfile, join
import os

optimize_params = True


def main(*args):
    print 'Running with following args \n'
    print args
    named_grid = namedtuple('Grid', ['n_comp', 'C', 'kernel', 'band_list', 'oacl_ranges', 'm'])
    search.grid = named_grid(*args)
    runs, evals = '', ''

    old_path = os.getcwd()
    os.chdir('..')

    oacl_ranges = search.grid.oacl_ranges
    pickel_file_name = str(oacl_ranges[0][0]) + str(oacl_ranges[0][1]) + str(oacl_ranges[1][0])
    pickel_file_name += str(oacl_ranges[1][1]) + str(search.grid.m) + '.dump'
    onlyfiles = [f for f in listdir('pickelfiles') if isfile(join('pickelfiles', f))]
    if len(onlyfiles) >= 100:
        file_to_delete1 = onlyfiles[0].replace('evals', '')
        file_to_delete1 = file_to_delete1.replace('runs', '')
        file_to_delete2 = 'runs' + file_to_delete1
        file_to_delete1 = 'evals' + file_to_delete1
        os.remove('pickelfiles/' + file_to_delete1)
        os.remove('pickelfiles/' + file_to_delete2)
    if 'runs' + pickel_file_name not in onlyfiles:
        with timed_block('Iteration '):
            runs = load_data(7, "T")
            eog_test, runs = extract_eog(runs)
            #runs, train_oacl = remake_trial(runs)

            evals = load_data(7, "E")
            eog_eval, evals = extract_eog(evals)
            #evals, test_oacl = remake_trial(evals, arg_oacl=train_oacl)

        # Save data, could be a method instead
        with open('pickelfiles/runs' + pickel_file_name, "wb") as output:
            cPickle.dump(runs, output, cPickle.HIGHEST_PROTOCOL)

        with open('pickelfiles/evals' + pickel_file_name, "wb") as output:
            cPickle.dump(evals, output, cPickle.HIGHEST_PROTOCOL)
    else:
        with open('pickelfiles/runs' + pickel_file_name, "rb") as input:
            runs = cPickle.load(input)

        with open('pickelfiles/evals' + pickel_file_name, "rb") as input:
            evals = cPickle.load(input)
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
        for band in train_bands:
            csp_list.append(csp_one_vs_all(band, 4))

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
    return score, 1200

if __name__ == '__main__':
    # freeze_support()
    main(2, 0.1, 'rbf', [[4, 8], [8, 12], [12, 16], [16, 20], [20, 30]], ((3, 7), (7, 15)), 11)
