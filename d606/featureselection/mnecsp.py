import mne
from mne.decoding import CSP
from preprocessing.dataextractor import *


def run_csp(run_data, label, n_comp):
    # transform data
    d3_data, labels = run_data

    labels = csp_label_reformat(labels, label)
    csp = CSP(n_components=n_comp)
    csp = csp.fit(array(d3_data), labels)
    return csp


def csp_one_vs_all(band_data, num_labels, n_comps=3):
    csp_list = []
    for n in range(1, num_labels + 1):
        csp = run_csp(band_data, n, n_comp=n_comps)
        csp_list.append(csp)

    return csp_list
