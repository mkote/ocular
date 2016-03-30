import mne
from mne.decoding import CSP
from d606.preprocessing.dataextractor import *


def run_csp(run_data, label):
    # transform data
    matrix, trials, labels = run_data
    d3_data = d3_matrix_creator(matrix)

    # create data info object for rawArray
    ch_names = ['eog' + str(x) for x in range(1, 4)]
    ch_names += ['eeg' + str(x) for x in range(1, 23)]

    ch_types = ['eog' for x in range(0, 3)]
    ch_types += ['eeg' for x in range(1, 23)]

    # Create label info
    labels = csp_label_reformat(labels, label)

    # Create data_info and event_info
    data_info = mne.create_info(ch_names, HERTZ, ch_types, None)
    event_info = create_events(labels)

    # Create mne structure
    epochs_data = mne.EpochsArray(d3_data, data_info, event_info)

    """ Do some crazy csp stuff """

    # Cross validation with sklearn
    labels = epochs_data.events[:, -1]

    n_components = 6  # pick some components
    csp = CSP(n_components=n_components)
    csp = csp.fit(d3_data, labels)
    return csp


def csp_one_vs_all(band_data, num_labels):
    csp_list = []
    for n in range(1, num_labels + 1):
        csp = run_csp(band_data, n)
        csp_list.append(csp)

    return csp_list
