from itertools import chain
import numpy as np
from copy import deepcopy


def create_mifs_list(selector, feature_vector_list, num_bands, num_components, labels):
    mifs_list = []

    for i in range(len(feature_vector_list)):
        # TODO: figure out which method should be used
        MIFS = deepcopy(selector)
        MIFS.fit(feature_vector_list[i], binarize_labels(labels, i))

        # Include all components of each CSP where at least one of its components has been selected
        selection = MIFS.support_
        b = num_bands
        m = num_components
        temp = [selection[j:j + m] for j in range(0, m * b, m)]
        temp2 = [[True] * m if True in temp[j] else [False] * m for j in range(b)]
        MIFS.support_ = np.array(list(chain(*temp2)))

        mifs_list.append(MIFS)

    return mifs_list


def binarize_labels(labels, i):
    # i is the class, i.e. the O in OVR
    minl = min(labels)
    return np.array([0 if i == j - minl else 1 for j in labels])
