from d606.preprocessing.dataextractor import load_data, restructure_data, extract_eog
from d606.preprocessing.filter import Filter
from d606.featureselection.mnecsp import csp_one_vs_all
from d606.classification.svm import csv_one_versus_all, svm_prediction
from d606.evaluation.timing import timed_block
from d606.evaluation.voting import csp_voting
from d606.evaluation.score import scoring
from d606.preprocessing.searchgrid import grid_combinator, grid_parameters, save_results
import d606.preprocessing.searchgrid as search
from d606.preprocessing.trial_remaker import remake_trial
from collections import namedtuple
from multiprocessing import freeze_support
from numpy import mean
import warnings


def main():
    Grid = namedtuple('Grid', ['band_list', 'n_comp', 'kernel', 'C'])
    best_result = [0, 0]

    warnings.filterwarnings('ignore', category=DeprecationWarning)
    warnings.filterwarnings('ignore', category=RuntimeWarning)

    runs = load_data(5, "T")
    eog_test, runs = extract_eog(runs)
    runs = remake_trial(runs)

    evals = load_data(5, "E")
    eog_eval, evals = extract_eog(runs)
    evals = remake_trial(evals)

    grid_list = grid_combinator(grid_parameters)

    for sample in grid_list:
        search.grid = Grid(*sample)
        subject_results = []

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
            subject_results.append(score)

            # store result in txt file
            save_results(score, search.grid)

            if score > best_result[0]:
                best_result[0] = score
                best_result[1] = search.grid

        print subject_results
        print search.grid
        print '\n'

    print "\t\tbest result: %s \n\t\tbest Parameters: %s" % (best_result[0], best_result[1])
    # print mean(subject_results)

if __name__ == '__main__':
    freeze_support()
    main()
