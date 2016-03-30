from sklearn.svm import SVC
from sklearn.cross_validation import ShuffleSplit
from d606.preprocessing.dataextractor import d3_matrix_creator, csp_label_reformat
from numpy import array, transpose


def train_svc(shuffle, csp, data, labels):
    cv = shuffle
    svc = SVC(C=1, kernel='linear')
    for train_idx, test_idx in cv:
        y_train, y_test = labels[train_idx], labels[test_idx]

        X_train = csp.transform(data[train_idx])

        # fit classifier
        svc.fit(X_train, y_train)

    return svc


def csv_one_versus_all(csp_list, band):
    data, trials, labels = band
    d3_data = d3_matrix_creator(data)
    svc_list = []
    cv = ShuffleSplit(len(labels), 10, test_size=0.2, random_state=42)
    for i, csp in enumerate(csp_list):
        formatted_labels = array(csp_label_reformat(labels, i+1))
        svc_list.append(train_svc(cv, csp, d3_data, formatted_labels))
    return svc_list
