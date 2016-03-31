from d606.preprocessing.dataextractor import load_data, dostuff
from d606.preprocessing.filter import Filter
from d606.featureselection.mnecsp import d3_matrix_creator, csp_one_vs_all
from d606.classification.svm import csv_one_versus_all
from d606.evaluation.voting import csp_voting
from d606.evaluation.score import scoring
from numpy import array, mean


subject_results = []
for subject in [int(x) for x in range(1, 2)]:
    csp_list = []
    svc_list = []

    runs = load_data(subject, "T")
    evals = load_data(subject, "E")

    filters = Filter([[6, 10], [10, 14], [14, 18],
                      [18, 22], [22, 26], [26, 30]])

    train_bands, train_combined_labels = dostuff(runs, filters)
    test_bands, test_combined_labels = dostuff(evals, filters)

    # CSP one VS all, give csp_one_cs_all num_different labels as input
    for band in train_bands:
        csp_list.append(csp_one_vs_all(band, 4))

    # Create a scv for each csp and band
    for csp, band in zip(csp_list, train_bands):
        svc_list.append(csv_one_versus_all(csp, band))

    single_run_result = []
    band_results = []
    results = []
    for y in range(0, len(test_bands)):
        d3_matrix = d3_matrix_creator(test_bands[y][0])
        for x in d3_matrix:
            for svc, csp in zip(svc_list[y], csp_list[y]):
                transformed = csp.transform(array([x]))
                single_run_result.append(int(svc.predict(transformed)))
            band_results.append(single_run_result)
            single_run_result = []
        results.append(band_results)
        band_results = []
    voting_results = csp_voting(results)
    score, wrong_list = scoring(voting_results, train_combined_labels)
    subject_results.append(score)
print subject_results
print mean(subject_results)
