from d606.preprocessing.dataextractor import *
from d606.preprocessing.filter import Filter


filt = Filter([[8, 12], [12, 24]])
f = load_data(1, "T")
temp = filt.filter(f[3])
data, trials, labels, artifacts = f[3]
matrix, trials = extract_trials(data, trials)
print "tro"
