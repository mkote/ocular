from d606.preprocessing.dataextractor import load_data, restructure_data
from d606.preprocessing.filter import Filter
from d606.featureselection.mnecsp import csp_one_vs_all
from d606.classification.svm import csv_one_versus_all, svm_prediction
from d606.evaluation.timing import timed_block
from d606.evaluation.voting import csp_voting
from d606.evaluation.score import scoring
from d606.preprocessing.searchgrid import grid_combinator, grid_parameters
import d606.preprocessing.searchgrid as search
from collections import namedtuple
from numpy import mean
import warnings

Grid = namedtuple('Grid', ['band_list'])

warnings.filterwarnings('ignore', category=DeprecationWarning)
warnings.filterwarnings('ignore', category=RuntimeWarning)

grid_list = grid_combinator(grid_parameters)

for sample in grid_list:
    search.grid = Grid(*sample)
    subject_results = []

    with timed_block('All Time'):
        for subject in [int(x) for x in range(1, 2)]:
            csp_list = []
            svc_list = []
            filt = search.grid.band_list if 'band_list' in search.grid._fields else [[8, 12], [16, 24]]
            filters = Filter(filt)

            runs = load_data(subject, "T")

            evals = load_data(subject, "E")

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

    print subject_results
    print search.grid
    print '\n'
    # print mean(subject_results)
