from d606.preprocessing.dataextractor import load_data, restructure_data, extract_eog
from d606.preprocessing.filter import Filter
from d606.featureselection.mnecsp import csp_one_vs_all
from d606.classification.svm import csv_one_versus_all, svm_prediction
from d606.evaluation.timing import timed_block
from d606.evaluation.voting import csp_voting
from d606.evaluation.score import scoring
from d606.preprocessing.searchgrid import grid_combinator, grid_parameters, save_results
import d606.preprocessing.searchgrid as search
from d606.preprocessing.trial_remaker import remake_trial, remake_single_trial
from collections import namedtuple
from multiprocessing import freeze_support
from itertools import product
import warnings


def main():
    # parameter C for svm
    Cs = [0.25, 0.50, 0.75, 1]

    # kernel for svm
    kernels = ['rbf', 'linear']

    # num components for csp
    n_comps = [3, 4, 5, 6, 7]

    # ranges for bandpassing
    band_lists = [[[10, 16], [16,28]],
                 [[8, 12], [16, 24]],
                 [[8, 12], [12, 16], [16, 20], [20, 24], [24, 28]],
                 [[4, 8], [8, 12], [12, 16], [16, 20], [20, 24], [24, 30], [30, 36]],
                 [[16, 20], [20, 24], [24, 30], [30, 36]]]

    for subject in range(1, 10):
        runs = load_data(subject, "T")
        eog_test, runs = extract_eog(runs)
        runs, train_oacl = remake_trial(runs)

        evals = load_data(subject, "E")
        eog_eval, evals = extract_eog(evals)
        evals, test_oacl = remake_trial(evals, arg_oacl=train_oacl)

        best_result = [0.0, "none"]
        subject_results = []

        with timed_block('All Time'):
            for band_list in band_lists:
                filt = band_list
                filters = Filter(filt)

                train_bands, train_combined_labels = restructure_data(runs, filters)

                test_bands, test_combined_labels = restructure_data(evals, filters)

                for comp in n_comps:
                    csp_list = []
                    # CSP one VS all, give csp_one_cs_all num_different labels as input
                    for band in train_bands:
                        csp_list.append(csp_one_vs_all(band, 4, n_comps=comp))

                    for kernel, c in product(kernels, Cs):
                        svc_list = []
                        # Create a scv for each csp and band
                        for csp, band in zip(csp_list, train_bands):
                            svc_list.append(csv_one_versus_all(csp, band, kernels=kernel, C=c))

                        # Predict results with svc's
                        results = svm_prediction(test_bands, svc_list, csp_list)

                        voting_results = csp_voting(results)
                        score, wrong_list = scoring(voting_results, test_combined_labels)
                        subject_results.append(score)
                        config = "band_list: " + str(band_list) + ", n_comp: " + str(comp) + ", kernel: " + str(kernel) + ", C: " + str(c)

                        if score > best_result[0]:
                            best_result[0] = score
                            best_result[1] = config

                        save_results(score, config)
                        print score
                        print "band_list: " + str(band_list) + ", n_comp: " + str(comp) + ", kernel: " + str(kernel) + ", C: " + str(c)
                        print '\n'

        save_results("Subject " + str(subject) + "best result: " + str(best_result[0]) + "\n",
                     str(best_result[1])+"\n\n")
        print "\t\tbest result: %s \n\t\tbest Parameters: %s" % (best_result[0], best_result[1])
        # print mean(subject_results)

if __name__ == '__main__':
    freeze_support()
    main()
