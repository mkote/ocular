import mne
from mne.decoding import CSP
from d606.preprocessing.dataextractor import *
from sklearn.pipeline import Pipeline  # noqa
from sklearn.cross_validation import cross_val_score  # noqa
from sklearn.svm import SVC  # noqa
from sklearn.cross_validation import ShuffleSplit  # noqa

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

# Create label info
labels = csp_label_reformat(labels, 2)

# Create data_info and event_info
data_info = mne.create_info(ch_names, HERTZ, ch_types, None)
event_info = create_events(labels)

# Create mne structure
epochs_data = mne.EpochsArray(d3_data, data_info, event_info)

""" Do some crazy csp stuff """

# Cross validation with sklearn
labels = epochs_data.events[:, -1]
evoked = epochs_data.average()

n_components = 3  # pick some components
svc = SVC(C=1, kernel='linear')
csp = CSP(n_components=n_components)

# Define a monte-carlo cross-validation generator (reduce variance):
cv = ShuffleSplit(len(labels), 10, test_size=0.2, random_state=42)
epochs_extract_data = epochs_data.get_data()

clf = Pipeline([('CSP', csp), ('SVC', svc)])
scores = cross_val_score(clf, epochs_extract_data, labels, cv=cv, n_jobs=1)
print(scores.mean())  # should match results above
