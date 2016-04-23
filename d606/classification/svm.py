from sklearn.svm import SVC
from preprocessing.dataextractor import d3_matrix_creator, csp_label_reformat
import preprocessing.searchgrid as search
from numpy import array


def train_svc(csp, data, labels):
    c = search.grid.C if 'C' in search.grid._fields else 1
    kernel = search.grid.kernel if 'kernel' in search.grid._fields else 'linear'
    svc = SVC(C=c, kernel=kernel, gamma='auto')

    y = labels
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


def svm_prediction(test_bands, svc_list, csp_list):
    # Lists to hold results
    single_run_result = []
    band_results = []
    results = []

    for y in range(0, len(test_bands)):
        d3_matrix = d3_matrix_creator(test_bands[y][0])
        for x in d3_matrix:
            for svc, csp in zip(svc_list[y], csp_list[y]):
                transformed = csp.transform(array([x]))
                single_run_result.append(int(svc.predict(transformed)))

            band_results.append(single_run_result)
            single_run_result = []

        results.append(band_results)
        band_results = []

    return results
