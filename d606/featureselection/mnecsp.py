import mne
from mne.decoding import CSP
from d606.preprocessing.dataextractor import *


# load data
data = load_data(1, 'T')
matrix, trials, labels, artifacts = data[3]

# transform data
matrix, trials = extract_trials(matrix, trials)
d3_data = d3_matrix_creator(matrix)

# create data info object for rawArray
ch_names = ['eog' + str(x) for x in range(1, 4)]
ch_names += ['eeg' + str(x) for x in range(1, 23)]

ch_types = ['eog' for x in range(0, 3)]
ch_types += ['eeg' for x in range(1, 23)]

# Create data_info and event_info
data_info = mne.create_info(ch_names, HERTZ, ch_types, None)
event_info = create_events(trials, labels)

# Create mne structure
epochs_data = mne.EpochsArray(d3_data, data_info, event_info)

csp = CSP(n_components=4)
csp.fit(d3_data, labels)
