from preprocessing.dataextractor import load_data, separate_eog_eeg, extract_real_trials
from preprocessing.oacl import moving_avg_filter, find_relative_heights, find_peak_indexes, find_artifact_signal
from multiprocessing import Array, Pool, cpu_count
from ctypes import c_float
from numpy import frombuffer, array
from itertools import product
import time


def init(trials_array):
    global shared_trials
    shared_trials = trials_array


def func(oacl_range, k, m):
    train = load_data(1, "T")
    train_eog, train_eeg = separate_eog_eeg(train)

    first_index = (train_eeg.index(x) for x in train_eeg if len(x[1]) > 0).next()

    train_flat_size = sum([x[0].shape[0] * x[0].shape[1] for x in train_eeg[first_index:]])
    num_channels = train_eeg[first_index][0].shape[0]
    channel_length = train_eeg[first_index][0].shape[1]
    run_length = num_channels * channel_length
    num_runs = len(train_eeg[first_index:])

    shared_trials_base = Array(c_float, train_flat_size, lock=False)
    shared_trials_array = frombuffer(shared_trials_base, dtype=c_float)

    for i, x in enumerate(train_eeg[first_index:]):
        for j, y in enumerate(x[0]):
            start = i * run_length + j * channel_length
            end = i * run_length + (j+1) * channel_length
            shared_trials_array[start:end] = y.flat

    init(shared_trials_array)

    pool = Pool(processes=cpu_count())
    artifact_signals = pool.map(gen_artifact_signal, [(x[0] * run_length + x[1] * channel_length,
                                                       x[0] * run_length + (x[1]+1) * channel_length,
                                                       x[1], m, oacl_range, k) for x
                                                      in product(range(0, num_runs), range(0, num_channels))])

    pool.close()
    pool.join()

    all_arti = []
    for x in range(0, num_runs):
        start = x * num_channels
        end = (x+1) * num_channels
        all_arti.append(artifact_signals[start:end])

    err = 0
    total_art = 0
    for x in range(0, num_runs):
        idx = first_index + x
        err += check_if_artifact(array(all_arti[x]), train_eeg[idx][1], train_eeg[idx][3])
        total_art += sum(train_eeg[idx][3])

    if err == total_art:
        err += 200

    return err, time.time()


def scale_range(rng, k, idx):
    l, u = rng
    layer = get_layer(idx)
    return l-(layer*k), u-(layer*k)


def get_layer(val):
    if val in [0]:
        return 0
    elif val in range(1, 6):
        return 1
    elif val in range(6, 13):
        return 2
    elif val in range(13, 18):
        return 3
    elif val in range(18, 21):
        return 4
    elif val in [21]:
        return 5


def gen_artifact_signal(args):
    start, end, chn_idx, m, rng, k = args
    scaled_range = scale_range(rng, k, chn_idx)
    padding = [0.0] * ((m-1)/2)
    filtered = moving_avg_filter(shared_trials[start:end], m)
    rh, z_idx = find_relative_heights(filtered)
    pk_idx = find_peak_indexes(rh, scaled_range)
    arti = find_artifact_signal(pk_idx, filtered, z_idx)
    return padding + arti + padding


def check_if_artifact(artifact_signals, trial_list, artifact_list):
    trials = extract_real_trials(artifact_signals, trial_list)
    actual_arti = sum([x for x in artifact_list])
    not_arti = len(artifact_list) - actual_arti
    err_arti = 0
    found_arti = 0
    for (y, z) in zip(trials, artifact_list):
        arti = contains_arti(y)
        if z == 0 and arti is True:
            err_arti += 1
        elif z == 0 and arti is False:
            continue
        elif z == 1 and arti is True:
            found_arti += 1
        elif z == 1 and arti is False:
            continue
    return (actual_arti - found_arti) + err_arti


def contains_arti(trial):
    temp = []
    for x in range(0, len(trial)):
        temp.append(all(v == 0.0 for v in trial[0]))
    return all(t is False for t in temp)

if __name__ == '__main__':
    func((2, 3), 0.0, 7)
