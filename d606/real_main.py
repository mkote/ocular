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


def main(*args):
    named_grid = namedtuple('Grid', ['n_comp', 'C'])

    runs = load_data(5, "T")
    eog_test, runs = extract_eog(runs)
    runs, train_oacl = remake_trial(runs)

    evals = load_data(5, "E")
    eog_eval, evals = extract_eog(evals)
    evals, test_oacl = remake_trial(evals, arg_oacl=train_oacl)

    search.grid = named_grid(*[args])

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
    return score

if __name__ == '__main__':
    freeze_support()
    main()
