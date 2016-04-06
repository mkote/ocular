# This module contains implementation of ocular artifact detection and removal

from math import exp, log
from decimal import Decimal, getcontext
from scipy.optimize import minimize
from d606.preprocessing.dataextractor import load_data, extract_trials_two
import numpy as np
import matplotlib.pyplot as plt

getcontext().prec = 300


def moving_avg_filter(chan_signal, m):
    # This function calculates the moving average filter of a single signal.
    # TODO: paper just uses symmetric MAF, but this loses information at ends!
    print "Applying moving average filter..."
    maf = []
    # We compute only so at each side of a point, we have (m-1)/2 elements.
    start = (m-1)/2
    end = len(chan_signal) - ((m-1) / 2)
    for i in range(start, end):
        ma_point = symmetric_moving_avg(chan_signal, m, i)
        maf.append(ma_point)
    return maf


def symmetric_moving_avg(chan_signal, m, t):
    """
    :param chan_signal: array of eeg time samples
    :param t: offset of sample to start with. Default is 0.
    :param m: number of neighboring points used in the filter.
    :return: smoothed signal.
    """
    n = len(chan_signal)
    if t + ((m - 1) / 2) >= n or t - (m / 2) < 0:
        raise ValueError("The desired range m for symmetric moving average "
                         "must be less for the given point t ")
    result = 0
    r = (m-1)/2
    for j in range(-r, r+1):
        val = chan_signal[t+j]
        result += val
    result *= 1 / float(m)
    return result


def find_relative_height(smooth_signal, time):
    """
    This function computes the relative height of points centered around
    point at param time.
    points
    :param smooth_signal: A list of (moving avg) points.
    :param time: the index in smoothed_data to compute the relative height on.
    :return: The relative height between time, time-1 and time+1
    """
    if time < 1:
        raise ValueError("Relative height undefined for index less than 1")

    # Here we subtract 2 because -1 is the last element and we want the
    # specified time to have 1 element on each side in the list.
    elif time > len(smooth_signal)-2:
        raise ValueError("Relative height undefined for index at the end of "
                         "the time series.")
    t = time
    x = smooth_signal
    diff1 = abs(x[t] - x[t-1])
    diff2 = abs(x[t+1] - x[t])
    rel_height = max(diff1, diff2)
    return rel_height


def find_relative_heights(smooth_signal):
    print "Finding relative heights..."
    relative_heights = []
    for i in range(1, len(smooth_signal)-1):
        next_rel_height = find_relative_height(smooth_signal, i)
        relative_heights.append(next_rel_height)
    return relative_heights


def find_peak_indexes(relative_heights, peak_range):
    print "finding peak indexes..."
    l = peak_range[0]
    u = peak_range[1]
    rh = relative_heights

    # Add 1 because index of rel heights list is one less than the
    # index of the smoothed data.
    Pt = [i+1 for i in range(0, len(rh)) if l < rh[i] < u]
    return Pt


def find_artifact_ranges(smooth_signal, peak_indexes):
    ranges = []
    for peak in peak_indexes:
        artifact_range = find_artifact_range(smooth_signal, peak)
        ranges.append(artifact_range)
    return ranges


def find_artifact_range(signal_smooth, peak):
    left = signal_smooth[0:peak]
    left.reverse()
    right = signal_smooth[peak+1:len(signal_smooth)]
    before_i = nearest_zero_point(left, 0, 1)
    after_i = nearest_zero_point(right, 0, 1)
    before = (peak-1) - before_i
    after = (peak+1) + after_i
    return before, after


def find_artifact_signal(peak_indexes, smooth_signal):
    artifact_signal = []
    ranges = find_artifact_ranges(smooth_signal, peak_indexes)
    for t in range(0, len(smooth_signal)):
        is_artifact = False
        for r in ranges:
            nzp_b = r[0]
            nzp_a = r[1]
            if nzp_b < t < nzp_a:
                is_artifact = True
                artifact = smooth_signal[t]
                artifact_signal.append(artifact)
                break
        if not is_artifact:
            artifact_signal.append(0.0)
    return artifact_signal

def find_artifact_signals(raw_signal, m, range_list):
    artifact_signals = []
    smooth_signal = moving_avg_filter(raw_signal, m)
    rh = find_relative_heights(smooth_signal)
    num_samples = len(raw_signal)
    i = 1
    for range in range_list:
        print "Processing signal for range: ", i
        peaks = find_peak_indexes(rh, range)
        artifact_signal = find_artifact_signal(peaks, smooth_signal)
        artifact_signals.append(artifact_signal)
        i += 1
    return np.array(artifact_signals)

def nearest_zero_point(arr, a, b):
    if b >= len(arr):
        # No polarity changes. So we just go with last element in the list.
        return a
    x = arr[a]
    y = arr[b]
    if x == 0.0:
        return a

    elif is_cross_zero(x, y):
        m = min(abs(x), abs(y))
        min_index = (a if m == arr[a] else b)
        return min_index
    else:
        return nearest_zero_point(arr, b, b + 1)


def is_cross_zero(a, b):
    if (a > 0 and b < 0) or (a < 0 and b > 0):
        return True
    else:
        return False


def learn_filtering_parameter():
    pass


def covariance_matrix(artifact_signal, raw_signal):
    return np.cov(m=artifact_signal, y=raw_signal)


def correlation_vector(artifact_signals, signal):
    return artifact_signals * signal.transpose()


def latent_var(theta, matrix, signal, b):
    pass


def logistic_function(z):
    if(z < -700):
        return 1
    return Decimal(1.0)/(Decimal(1.0)+Decimal(exp(-z)))


def column(matrix, i):
    return [row[i] for row in matrix]


def variance(vector):
    avg = int(np.mean(vector))
    return sum([pow(val - avg, 2) for val in vector]) / len(vector)


def objective_function(theta, b, labels, n_trials, trial_artifact_signals,
                       trial_signals):
    summa = 0
    thetaT = theta.transpose()
    y = labels

    for i in range(n_trials):
        Xa = np.mat(column(trial_artifact_signals, i))
        x0 = trial_signals[i]

        k1 = thetaT * (Xa * Xa.transpose()) * theta
        k2 = 2 * thetaT * (Xa * x0.transpose())

        z = float(k1 - k2 + variance(x0.tolist()[0]) + b)
        h = logistic_function(z)

        # hack
        if(h == 1):
            h -= Decimal(10)**(-300)

        summa += -y[i] * log(h, 2) - (1 - y[i]) * log(1 - h, 2);

    return summa / n_trials


def remove_ocular_artifacts(raw_signal, theta, artifact_signals):
    A = theta.transpose().dot(artifact_signals).transpose()
    corrected_signal = [a_i - b_i for a_i, b_i in zip(raw_signal, A)]
    return np.array(corrected_signal)


def objective_function_aux(args, args2):
    arg1 = np.array([[args[k]] for k in xrange(len(args)-1)])
    arg2 = args[len(args)-1]
    return objective_function(arg1, arg2, *args2)


def plot_example():
    eeg_data = load_data(1, 'T')
    raw_signal = eeg_data[5][0][4]
    raw_signal_eog = eeg_data[5][0][23]
    raw_signal = raw_signal[0:2000]
    raw_signal_eog = raw_signal_eog[0:2000]
    labels = eeg_data[5][2]
    n_trials = 48
    m = 11
    signal = raw_signal
    matrix = np.cov(raw_signal)

    plt.axis([0, len(raw_signal), min(raw_signal), max(raw_signal)])
    plt.ylabel('amplitude')
    plt.xlabel('time point')
    plt.figure(1)
    plt.subplot(211)
    plt.plot([x for x in range(0, len(raw_signal))], raw_signal)
    plt.subplot(211)
    m = 11
    num_samples = len(raw_signal)
    filtered_signal = moving_avg_filter(raw_signal, m)
    plt.plot([x for x in range(0, len(filtered_signal))], filtered_signal)

    plt.subplot(211)
    rh = find_relative_heights(filtered_signal)
    ti = find_peak_indexes(rh, (8, 1.5))
    artifact_signal = find_artifact_signal(ti, filtered_signal)
    plt.plot([x for x in range(0, len(artifact_signal))], artifact_signal)

    plt.subplot(212)
    plt.plot([x for x in range(0, len(raw_signal_eog))], raw_signal_eog)
    plt.show()


def extract_trials_array(signal, trials_start):
    n_trials = len(trials_start)
    concat_trials, concat_trials_start = extract_trials_two(signal,
                                                       trials_start)
    trials = [concat_trials[concat_trials_start[i]:concat_trials_start[i+1]]
              for i in xrange(n_trials-1)]
    trials += [concat_trials[concat_trials_start[n_trials-1]:]]
    return trials


def clean_signal(raw_signal, trials_start, labels, range_list, m):
    n_trials = len(trials_start)              # Number of trials
    artifact_signals = find_artifact_signals(raw_signal, m, range_list)
    trial_signals = np.mat(extract_trials_array(raw_signal, trials_start))
    trial_artifact_signals = [extract_trials_array(artifact_signals[i], trials_start)
                              for i in xrange(len(range_list))]
    print("Minimizing...")
    min_result = minimize(objective_function_aux,
                          [0.5] * (len(range_list) + 1),
                          bounds=[[0, 1]] * (len(range_list) + 1),
                          method="L-BFGS-B",
                          args=[labels, n_trials, trial_artifact_signals,
                                trial_signals])
    filtering_param = np.array([[min_result.x[k]] for k in xrange(len(min_result.x) - 1)])
    # b = min_result.x[len(min_result.x) - 1]

    return remove_ocular_artifacts(raw_signal, filtering_param, artifact_signals)

def clean_eeg(eeg_data, range_list = [(4, 6), (8, 15)], run_index = 5, m = 11):
    channels, trials_start, labels, artifacts = eeg_data[run_index]
    raw_signals = channels[0:22]        # EEG channels (raw)
    clean_signals = []
    for raw_signal in raw_signals:
        clean_signals.append(clean_signal(raw_signal,
                                          trials_start,
                                          labels,
                                          range_list,
                                          m))
    return clean_signals
