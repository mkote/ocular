from d606.preprocessing.dataextractor import load_data, run_combiner, \
    extract_trials_two
from d606.preprocessing.filter import Filter
from d606.featureselection.mnecsp import d3_matrix_creator, csp_one_vs_all
from d606.classification.svm import csv_one_versus_all
from d606.evaluation.voting import csp_voting
from d606.evaluation.score import scoring
from numpy import array, mean


subject_results = []
for subject in [int(x) for x in range(1, 10)]:
    csp_list = []
    svc_list = []
    bands = []
    filter_bank = []
    data_list = []
    combined_data = []
    combined_labels = []

    runs = load_data(subject, "T")
    evals = load_data(subject, "E")

    filters = Filter([[6, 10], [10, 14], [14, 18], 
        [18, 22], [22, 26], [26, 30]])
    num_banks = len(filters.band_range)

    for run in runs:
        matrix, trials, labels, artifacts = run
        filter_bank.append(filters.filter(matrix))
        data_list.append((trials, labels, artifacts))
    del run, runs

    filt_tuples = [[] for x in range(0, len(filter_bank[0]))]
    data_tuple_bands = [[] for x in range(0, len(filter_bank[0]))]

    # Restructure matrices, and recreate data tuples
    for bank in filter_bank:
        for x in range(0, num_banks):
            filt_tuples[x].append(bank[x])
    del bank, filter_bank

    for i, data in enumerate(data_list):
        for x in range(0, num_banks):
            trials, labels, artifacts = data
            matrix = filt_tuples[x][i]
            data_tuple_bands[x].append((matrix, trials, labels, artifacts))
    del data, trials, labels, artifacts, matrix, data_list, filt_tuples

    # Call run_combiner with band from data_tuples
    for x in [0 for y in range(0, len(data_tuple_bands))]:
        combined_data.append(run_combiner(data_tuple_bands[x]))
        del data_tuple_bands[x]

    # Trial Extraction before csp and svn
    for eeg_signal in combined_data:
        old_matrix, old_trials, labels, artifacts = eeg_signal
        new_matrix, new_trials = extract_trials_two(old_matrix, old_trials)
        bands.append((new_matrix, new_trials, labels))
    combined_labels.extend(combined_data[0][2])
    del combined_data
    
    # CSP one VS all, give csp_one_cs_all num_different labels as input
    for band in bands:
        csp_list.append(csp_one_vs_all(band, 4))

    # Create a scv for each csp and band
    for csp, band in zip(csp_list, bands):
        svc_list.append(csv_one_versus_all(csp, band))

    single_run_result = []
    band_results = []
    results = []
    for y in range(0, len(bands)):
        d3_matrix = d3_matrix_creator(bands[y][0])
        for x in d3_matrix:
            for svc, csp in zip(svc_list[y], csp_list[y]):
                transformed = csp.transform(array([x]))
                single_run_result.append(int(svc.predict(transformed)))
            band_results.append(single_run_result)
            single_run_result = []
        results.append(band_results)
        band_results = []
    voting_results = csp_voting(results)
    score, wrong_list = scoring(voting_results, combined_labels)
    subject_results.append(score)
print subject_results
print mean(subject_results)
