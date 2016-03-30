# This module contains implementation of ocular artifact detection and removal
from math import exp, log
from scipy.optimize import minimize

from d606.preprocessing.dataextractor import load_data
from numpy import transpose, ndarray


def moving_avg_filter(chan_signal, m):
    # This function calculates the moving average filter of a single signal.
    # TODO: paper just uses symmetric MAF, but this loses information at ends!
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


def relative_height(smooth_signal, time):
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
    relative_heights = []
    for i in range(1, len(smooth_signal)-1):
        next_rel_height = relative_height(smooth_signal, i)
        relative_heights.append(next_rel_height)
    return relative_heights


def find_time_indexes(relative_heights, peak_range, m_range, n_samples):
    l = peak_range[0]
    u = peak_range[1]
    rh = relative_heights
    time_indexes_in_range = []

    for i in range(0, len(relative_heights)):
        if m_range/2 < i < n_samples-(m_range/2) and l < rh[i] < u:
            # Add 1 because index of rel heights list is one less than the
            # index of the smoothed data.
            time_indexes_in_range.append(i+1)
    return time_indexes_in_range


def artifact_ranges(smooth_signal, peak_indexes):
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


def artifact_signals(peak_indexes, smooth_signal):
    artifact_signal = []
    ranges = artifact_ranges(smooth_signal, peak_indexes)
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
    return cov(m=artifact_signal, y=raw_signal)


def variance():
    pass


def correlation_vector(artifact_signals, signal):
    return artifact_signals * signal.transpose()


def latent_var(theta, matrix, signal, b):
    global artifact_signal
    Ra = cov(matrix)
    ra = correlation_vector(artifact_signal, signal)
    r0 = variance(signal)
    result = theta.transpose() * Ra * theta - 2*theta.transpose()* ra + r0 + b
    return result

def cov(matrix):
    return matrix*matrix.transpose()


def logistic_function(latent_variable):
    return 1.0/(1.0+exp(-latent_variable))


def objective_function(theta, b):
    global labels
    global matrix
    global signal
    global n_trials
    y = labels
    z = latent_var(theta, matrix, signal, b)
    nt = n_trials
    result = 0
    for j in range(1, m):
        h = logistic_function(z)
        result += -y[j]*log(h, 2)-(1-y[j]*log(1-h, 2))
    result *= (1.0/nt) * result
    return result


eeg_data = load_data(1, 'T')
raw_signal = eeg_data[5][0][4]
raw_signal_eog = eeg_data[5][0][1]
raw_signal = raw_signal[0:2000]
raw_signal_eog = raw_signal_eog[0:2000]
labels = eeg_data[5][2]
n_trials = 48
m = 11
signal = raw_signal
matrix = cov(raw_signal)


import matplotlib.pyplot as plt

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
ti = find_time_indexes(rh, (8, 1.5 ), m, num_samples)
artifact_signal = artifact_signals(ti, filtered_signal)
plt.plot([x for x in range(0, len(artifact_signal))], artifact_signal)

plt.subplot(212)
plt.plot([x for x in range(0, len(raw_signal_eog))], raw_signal_eog)
plt.show()
x0 = [1, 2]
#  res = minimize(objective_function, x0, args=(-1.0,), method='SLSQP')
#  print(res.x)