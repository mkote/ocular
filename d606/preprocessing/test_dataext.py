from dataextractor import load_data, separate_eog_eeg
from filter import Filter
from itertools import chain
from multiprocessing import Pool, Array, cpu_count
from featureselection.mnecsp import csp_one_vs_all
from sklearn.ensemble import RandomForestClassifier
from skll import metrics
from numpy import array, mean, frombuffer
from trial_remaker import remake_trial
from oaclbase import OACL
from sklearn import cross_validation
from time import time
from ctypes import c_float, c_int


def create_trial_pool(runs, filters):
    bands = []
    combined_data = []
    combined_labels = []
    filter_bank = []
    num_banks = len(filters.band_range)

    for run in runs:
        matrix, trials, labels, artifacts = run
        filter_bank.append([(single_filter, trials, labels, artifacts) for single_filter in filters.filter(matrix)])

    #  Restructure matrices, and recreate data tuples
    pool = [[] for n in range(num_banks)]
    pool_labels = list(chain(*[run[2] for run in runs]))
    for bank in filter_bank:
        for x in range(0, num_banks):
            pool[x].extend(extract_trials(bank[x][0], bank[x][1]))
    del bank, filter_bank

    return array(pool), pool_labels


def extract_trials(data, timestamps):
    CUE = 700
    TRIAL_TIME = CUE + 750
    trials = [[list(chan[x+CUE : x+TRIAL_TIME]) for chan in data] for x in timestamps]
    return trials


def create_feature_vector_list(data, csps):
    temp = []
    for csp in csps:
        temp.append(csp.transform(array(data)))

    temp = [list(chain(*x)) for x in list(chain(zip(*temp)))]
    return temp


def extract_and_classify(args):
    train_idx, test_idx, band_idx, n_comp, trial_length, band_size, trial_size = args
    train = []
    test = []

    train_labels = label_data[train_idx]

    for t in train_idx:
        start = band_idx * band_size + t * trial_size
        end = start + trial_size
        train.append(array([train_data[start:end][h:l] for h,l in zip([h for h in range(0, end, trial_length)], [l for l in range(trial_length, (trial_size)+trial_length, trial_length)])]))

    for t in test_idx:
        start = band_idx * band_size + t * trial_size
        end = start + trial_size
        test.append(array([train_data[start:end][h:l] for h,l in zip([h for h in range(0, end, trial_length)], [l for l in range(trial_length, (trial_size)+trial_length, trial_length)])]))

    train_col = (train, train_labels)
    csps = (csp_one_vs_all(train_col, 4, n_comps=n_comp))

    train_features = create_feature_vector_list(train, csps)
    test_features = create_feature_vector_list(test, csps)

    return train_features, test_features


def init(train, labels):
    global train_data, label_data
    train_data = train
    label_data = labels


def randforest(train_features, train_labels, test_features, test_labels, n_comp, seeds=(4, 8, 15, 16, 23, 42)):
    kappas = []
    accuracies = []
    for seed in seeds:
        rf = RandomForestClassifier(n_estimators=4 * n_comp, random_state=seed)
        rf.fit(train_features, train_labels)

        predictions = []
        for y in test_features:
            predictions.append(rf.predict(array(y).reshape(1, -1)))

        kappa = metrics.kappa(test_labels, predictions)
        accuracy = mean([a == b for (a, b) in zip(predictions, test_labels)])

        kappas.append(kappa)
        accuracies.append(accuracy)

    return mean(accuracies), mean(kappas)


def main(n_comp, band_list, subject, oacl_ranges=None, m=None, thetas=None):
    TRIAL_LENGTH = 750
    train = load_data(subject, "T")
    _, train = separate_eog_eeg(train)
    oacl = OACL(ranges=oacl_ranges, m=m, multi_run=True, trials=False)
    oacl.theta = thetas
    train, _ = remake_trial(train, m=m, oacl_ranges=oacl_ranges, arg_oacl=oacl)
    filters = Filter(band_list)
    num_bands = len(filters.band_range)
    num_channels = len(train[0][0])
    train_pool, train_label_pool = create_trial_pool(train, filters)
    n_trials = len(train_label_pool)

    rs = cross_validation.ShuffleSplit(n_trials, n_iter=10, test_size=.25, random_state=0)

    train_flat_size = num_bands * n_trials * num_channels * TRIAL_LENGTH
    label_flat_size = len(train_label_pool)

    shared_train_base = Array(c_float, train_flat_size, lock=False)
    shared_train_array = frombuffer(shared_train_base, dtype=c_float)

    shared_label_base = Array(c_int, label_flat_size, lock=False)
    shared_label_array = frombuffer(shared_label_base, dtype=c_int)

    trial_size = num_channels * TRIAL_LENGTH
    for b in range(num_bands):
        for i, t in enumerate(train_pool[b]):
            v1 = i * trial_size
            v2 = (i + 1) * trial_size
            start = b * n_trials * trial_size + v1
            end = b * n_trials * trial_size + v2
            shared_train_array[start:end] = array(t.flat)

    shared_label_array[:] = array(array(train_label_pool).flat)

    init(shared_train_array, shared_label_array)

    accuracies = []
    kappas = []
    for train_idx, test_idx in rs:
        extract_and_classify([train_idx, test_idx, 0, 3, TRIAL_LENGTH,  n_trials * num_channels * TRIAL_LENGTH, num_channels * TRIAL_LENGTH])
        pool = Pool(processes=cpu_count())
        out = pool.map(extract_and_classify, [(train_idx, test_idx, band, n_comp, TRIAL_LENGTH,
                                               n_trials * num_channels * TRIAL_LENGTH, num_channels * TRIAL_LENGTH)
                                              for band in range(num_bands)], chunksize=1)
        pool.close()
        pool.join()

        train_features = [list(chain(*y)) for y in zip(*[x[0] for x in out])]
        test_features = [list(chain(*y)) for y in zip(*[x[1] for x in out])]
        train_labels = label_data[train_idx]
        test_labels = label_data[test_idx]
        acc, kap = randforest(train_features, train_labels, test_features, test_labels, n_comp)
        accuracies.append(acc)
        kappas.append(kap)

    print "Mean accuracy: " + str(mean(accuracies) * 100)
    print "Mean kappa: " + str(mean(kappas))
    return mean(accuracies) * 100, time()



if __name__ == '__main__':
    main(3, [[8, 12], [16, 24]], 1, ((2,3),), 7, [array([0.0]) for x in range(22)])
