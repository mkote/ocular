from d606.preprocessing.dataextractor import load_data, dostuff
from d606.preprocessing.filter import Filter
from d606.featureselection.mnecsp import d3_matrix_creator, csp_one_vs_all
from d606.classification.svm import csv_one_versus_all
from d606.evaluation.voting import csp_voting
from d606.evaluation.score import scoring
from numpy import array, mean
import time


subject_results = []
for subject in [int(x) for x in range(1, 10)]:
    csp_list = []
    svc_list = []
    r_s = time.time()
    runs = load_data(subject, "T")
    f_r_e = time.time()
    print "Trial data load time: ", f_r_e-r_s
    evals = load_data(subject, "E")
    s_r_e = time.time()
    print "Test data load time: ", s_r_e-f_r_e

    filters = Filter([[6, 10], [10, 14], [14, 18],
                      [18, 22], [22, 26], [26, 30]])
    tr_ex_s = time.time()
    train_bands, train_combined_labels = dostuff(runs, filters)
    tr_ex_e = time.time()
    print "Trial data extract time: ", tr_ex_e-tr_ex_s
    test_bands, test_combined_labels = dostuff(evals, filters)
    te_ex_e = time.time()
    print "Test data extract time: ", te_ex_e-tr_ex_e
    s_time = time.time()
    # CSP one VS all, give csp_one_cs_all num_different labels as input
    for band in train_bands:
        csp_list.append(csp_one_vs_all(band, 4))
    m_time = time.time()
    print "CSP takes: ", m_time-s_time
    # Create a scv for each csp and band
    for csp, band in zip(csp_list, train_bands):
        svc_list.append(csv_one_versus_all(csp, band))
    e_time = time.time()
    print "SVC takes: ", e_time-m_time
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
    to_t = time.time()
    print "Total time taken: ", to_t-r_s
print subject_results
print mean(subject_results)
