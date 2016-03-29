from d606.preprocessing.dataextractor import *
from d606.preprocessing.filter import Filter
from d606.featureselection.mnecsp import *


filt = Filter([[8, 12], [12, 24]])
f = load_data(1, "T")
data, trials, labels, artifacts = f[3]
banks = filt.filter(data)
bands = []
for eeg_signal in banks:
    matrix, new_trials = extract_trials(eeg_signal, trials)
    bands.append((matrix, new_trials, labels))

csp_list = csp_one_vs_all(bands[0], 4)
print "Kristian"
