# This module contains implementation of ocular artifact detection and removal
from decimal import *

from d606.preprocessing.dataextractor import load_data


def moving_avg_filter(chan_signal, m):
    # This function calculates the moving average filter of a single signal.
    # TODO: paper just uses symmetric MAF, but this loses information at ends!
    maf = []
    # We compute only so at each side of a point, we have (m-1)/2 elements.
    start = (m-1)/2
    sig_size = len(chan_signal)
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
    if t + ((m-1)/2) > len(chan_signal) or t-((m-1)/2) < 0:
        raise ValueError("The desired range m for symmetric moving average "
                         "must be less for the given point t ")
    result = 0
    r = (m-1)/2
    for j in range(-r, r+1):
        val = chan_signal[t+j]
        result += val
    result = result * (1 / float(m))
    return result


def relative_height(smoothed_data, time):
    t = time
    x = smoothed_data
    peak1 = abs(x[t] - x[t-1])
    peak2 = abs(x[t+1] - x[t])
    rel_height = max(peak1, peak2)
    return rel_height


def find_time_indexes(smoothed_data, relative_height, time, points_used, l,
                      u, r, c):
    """
    :param time: the time offset
    :param points: the number of points used in moving average
    :return: the set of time indexes of the peaks in range h
    :l: lower bound of peak heights.
    :u: upper bound of peak heights.
    :r: number of channels
    :c: number of samples
    """
    m = points_used
    t = time
    rh = relative_height
    sd = smoothed_data
    p = [t for t in sd and m/2 < t < c and l < rh < u]
    return p


def artifact_signals():
    pass


eeg_data = load_data(1, 'T')

raw_signal = eeg_data[3][0][0]
import matplotlib.pyplot as plt
plt.plot([x for x in range(0, len(raw_signal))], raw_signal)
plt.axis([0, len(raw_signal), min(raw_signal), max(raw_signal)])
plt.ylabel('amplitude')
plt.xlabel('time point')
plt.show()
filtered_signal = moving_avg_filter(raw_signal, 50)
plt.plot([x for x in range(0, len(filtered_signal))], filtered_signal)
plt.axis([0, len(filtered_signal), min(filtered_signal), max(filtered_signal)])
plt.ylabel('amplitude')
plt.xlabel('time point')
plt.show()