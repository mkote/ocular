from sklearn.svm import SVC
from sklearn.preprocessing import MinMaxScaler
from sklearn.cross_validation import ShuffleSplit
from d606.preprocessing.dataextractor import d3_matrix_creator, \
    csp_label_reformat
from numpy import array


def train_svc(csp, data, labels):
    svc = SVC(C=1, kernel='rbf', gamma='auto')

    y = labels
    min_max = MinMaxScaler()
    x = csp.transform(data)

    # fit classifier
    svc.fit(x, y)

    return svc


def csv_one_versus_all(csp_list, band):
    data, trials, labels = band
    d3_data = d3_matrix_creator(data)
    svc_list = []
    for i, csp in enumerate(csp_list):
        formatted_labels = array(csp_label_reformat(labels, i+1))
        svc_list.append(train_svc(csp, d3_data, formatted_labels))
    return svc_list
