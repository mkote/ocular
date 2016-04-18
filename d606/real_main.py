from preprocessing.dataextractor import load_data, restructure_data, extract_eog
from preprocessing.filter import Filter
from featureselection.mnecsp import csp_one_vs_all
from classification.svm import csv_one_versus_all, svm_prediction
from evaluation.timing import timed_block
from evaluation.voting import csp_voting
from evaluation.score import scoring
import preprocessing.searchgrid as search
from preprocessing.trial_remaker import remake_trial
from collections import namedtuple
from multiprocessing import freeze_support
import cPickle
import os

run_oacl = True


def main(*args):
    named_grid = namedtuple('Grid', ['n_comp', 'C', 'kernel', 'band_list', 'oacl_ranges', 'm'])
    print args
    search.grid = named_grid(*args)
    runs, evals = '', ''

    old_path = os.getcwd()
    if 'd606' not in old_path:
        os.chdir('../../../d606')

    if not os.path.isfile('runs.dump') and not os.path.isfile('evals.dump') or run_oacl is True:
        runs = load_data(1, "T")
        eog_test, runs = extract_eog(runs)
        runs, train_oacl = remake_trial(runs)

        evals = load_data(1, "E")
        eog_eval, evals = extract_eog(evals)
        evals, test_oacl = remake_trial(evals, arg_oacl=train_oacl)

        # Save data, could be a method instead
        if os.path.isfile('runs.dump'):
            os.remove('runs.dump')
        with open("runs.dump", "wb") as output:
            cPickle.dump(runs, output, cPickle.HIGHEST_PROTOCOL)

        if os.path.isfile('evals.dump'):
            os.remove('evals.dump')
        with open("evals.dump", "wb") as output:
            cPickle.dump(evals, output, cPickle.HIGHEST_PROTOCOL)
    else:
        with open("runs.dump", "rb") as input:
            runs = cPickle.load(input)

        with open("evals.dump", "rb") as input:
            evals = cPickle.load(input)

    os.chdir(old_path)

    with timed_block('All Time'):
        # for subject in [int(x) for x in range(1, 2)]:
        csp_list = []
        svc_list = []
        filt = search.grid.band_list if 'band_list' in search.grid._fields else [[8, 12], [16, 24]]
        filters = Filter(filt)

        train_bands, train_combined_labels = restructure_data(runs, filters)

        test_bands, test_combined_labels = restructure_data(evals, filters)

        # CSP one VS all, give csp_one_cs_all num_different labels as input
        for band in train_bands:
            csp_list.append(csp_one_vs_all(band, 4))

        # Create a scv for each csp and band
        for csp, band in zip(csp_list, train_bands):
            svc_list.append(csv_one_versus_all(csp, band))

        # Predict results with svc's
        results = svm_prediction(test_bands, svc_list, csp_list)

        voting_results = csp_voting(results)
        score, wrong_list = scoring(voting_results, test_combined_labels)

    print search.grid
    print '\n'
    print score
    return score

# if __name__ == '__main__':
#     freeze_support()
#     main(2, 0.1, 'rbf', [[10, 20], [12, 14]], ((3, 7), (7, 15)), 11)
