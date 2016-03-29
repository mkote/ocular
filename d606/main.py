from d606.preprocessing.dataextractor import load_data, run_combiner, \
    extract_trials
from d606.preprocessing.filter import Filter
from d606.featureselection.mnecsp import d3_matrix_creator, csp_one_vs_all
from d606.classification.svm import csv_one_versus_all
from numpy import array, empty

csp_list = []
svc_list = []
bands = []
banks = []

runs = load_data(1, "T")
filt = Filter([[8, 12], [16, 24]])
for run in runs:
    matrix, trials, labels, artifacts = run
    banks.append(filt.filter(matrix))

mod_run = [[] for x in range(0, len(banks[0]))]

for bank in banks:
    mod_run[0].append(bank[0])
    mod_run[1].append(bank[1])

temp = (mod_run[0], trials, labels, artifacts)

data, trials, labels, artifacts = run_combiner(temp)


for eeg_signal in banks:
    matrix, new_trials = extract_trials(eeg_signal, trials)
    bands.append((matrix, new_trials, labels))

for band in bands:
    csp_list.append(csp_one_vs_all(band, 4))

for csps, band in zip(csp_list, bands):
    svc_list.append(csv_one_versus_all(csps, band))

result = []
results = []
resultss = []
for y in range(0, len(bands)):
    d3_matrix = d3_matrix_creator(bands[y][0])
    for x in d3_matrix:
        for svc, csp in zip(svc_list[y], csp_list[y]):
            transformed = csp.transform(array([x]))
            result.append(svc.predict(transformed))
        results.append(result)
        result = []
    resultss.append(results)
    results = []
print "Kristian"
