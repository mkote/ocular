from d606.trialextractor2 import *
from mne.decoding import CSP
from mne import EpochsArray
from sklearn.svm import SVC
from sklearn.multiclass import OneVsRestClassifier
'''
    The framework takes .mat files as input. You have to specify which
    preproces-sing steps will be performed. Can be used from the terminal,
    but read documentation for a full description
'''

# Data to load
# eeg_data = load_data(1, 'T')

'''!!! Need to implement SOBI as an initial step here!!!'''

# Add noise removal techniques here. They must take as input all eeg_data
# and output all eeg_data again, to standardize input and output from
# function calls.
# noise_removal_methods = []

# for method in noise_removal_methods:
#    eeg_data = method(eeg_data)

# Here we run additional preprocessing, feature extraction, featureselection,
# and classification methods, before the evaluation of the pipeline begins.
# pipeline = [dwt1]

# for method in pipeline:
#    eeg_data = method(eeg_data)
#
# results = []
# for x in range(1, 10):
#     # Last parameter is the lower & higer band for bandpass
#     s = Subject("A0" + str(x), "T", [8, 30])
#
#     # Chose the labels you want to extract trials for
#     a = s.trials_with_label([1, 2])
#
#     epochs = EpochsArray(a.epochs, a.config, a.events)
#
#     labels = epochs.events[:, -1]
#
#     evoked = epochs.average()
#
#     n_components = 5  # pick some components
#     svc = SVC(C=0.67, kernel='linear')
#     csp = CSP(n_components=n_components)
#
#     # Define a monte-carlo cross-validation generator (reduce variance):
#     cv = ShuffleSplit(len(labels), 10, test_size=0.2, random_state=42)
#     scores = []
#     epochs_data1 = epochs.get_data()
#
#     for train_idx, test_idx in cv:
#         y_train, y_test = labels[train_idx], labels[test_idx]
#
#         X_train = prepros.scale(csp.fit_transform(epochs_data1[train_idx],
#                                                   y_train))
#         X_test = prepros.scale(csp.transform(epochs_data1[test_idx]))
#
#         svc.fit(X_train, y_train)
#
#         scores.append(svc.score(X_test, y_test))
#
#     # Printing the results
#     class_balance = np.mean(labels == labels[0])
#     class_balance = max(class_balance, 1. - class_balance)
#     print("Classification accuracy: %f / Chance level: %f" %
#                                                            (np.mean(scores),
#                                                            class_balance))
#     results.append([x, np.mean(scores)])
#
# _sum = []
# for x in results:
#     print(x)
#     _sum.append(x[1])
# print("\nAverage: " + str(np.mean(_sum)))

# Initialize subject
csp = CSP(n_components=8)
svc = SVC(C=0.67, kernel='linear')
s = Subject("1", "T")
test = []

# Extract one versus rest for all the classes
one_versus_rest = []
weights = []
for x in range(1, 5):
    one_versus_rest.append(s.trials_for_class_vs_rest(x))

# Run CSP on every of the above classes
i = 0
for x in one_versus_rest:
    epochsArr = EpochsArray(x.epochs, x.config, x.events)
    test.append(epochsArr.get_data().copy())
    weights.append(csp.fit_transform(epochsArr.get_data(),
                                     epochsArr.events[:, -1]))
    test[i] = csp.transform(test[i])
    i += 1


# Classify for each of the given CSPs
print("done")

# Vote for the most likely candidate
