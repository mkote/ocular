from d606.preprocessing.dataextractor import *
from d606.preprocessing.filter import Filter
from d606.featureselection.mnecsp import *
from d606.classification.svm import csv_one_versus_all
from numpy import array


filt = Filter([[8, 12], [12, 24]])
f = load_data(1, "T")
data, trials, labels, artifacts = f[3]
banks = filt.filter(data)
bands = []
for eeg_signal in banks:
    matrix, new_trials = extract_trials(eeg_signal, trials)
    bands.append((matrix, new_trials, labels))

csp_list = csp_one_vs_all(bands[0], 4)

svc_list = csv_one_versus_all(csp_list, bands[0])
temp = []
results = []
temp.append(bands[0][0][0:25, 1500:2250])
for svc, csp in zip(svc_list, csp_list):
    transformed = csp.transform(array(temp))
    results.append(svc.predict(transformed))
print "Kristian"
