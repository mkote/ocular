from sklearn.ensemble import RandomForestClassifier
from preprocessing.dataextractor import d3_matrix_creator, csp_label_reformat
from numpy import array


def train_rfl(csp, data, labels):
    clf = RandomForestClassifier(n_estimators=20, max_depth=None, min_samples_split=1, max_features='sqrt')
    y = labels
    x = csp.transform(data)

    # fit classifier
    clf.fit(x, y)

    return clf


def rfl_one_versus_all(csp_list, band):
    data, trials, labels = band
    d3_data = d3_matrix_creator(data)
    rfl_list = []
    for i, csp in enumerate(csp_list):
        formatted_labels = array(csp_label_reformat(labels, i+1))
        rfl_list.append(train_rfl(csp, d3_data, formatted_labels))
    return rfl_list


def rfl_prediction(test_bands, rfl_list, csp_list):
    # Lists to hold results
    single_run_result = []
    band_results = []
    results = []

    for y in range(0, len(test_bands)):
        d3_matrix = d3_matrix_creator(test_bands[y][0])
        for x in d3_matrix:
            for rfl, csp in zip(rfl_list[y], csp_list[y]):
                transformed = csp.transform(array([x]))
                single_run_result.append(int(rfl.predict(transformed)))

            band_results.append(single_run_result)
            single_run_result = []

        results.append(band_results)
        band_results = []

    return results
