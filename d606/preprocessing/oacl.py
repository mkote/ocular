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
    n = len(chan_signal)
    if t + ((m-1)/2) >= n or t-((m)/2) < 0:
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
    """
    This function computes the relative height of points centered around
    point at param time.
    points
    :param smoothed_data: A list of (moving avg) points.
    :param time: the index in smoothed_data to compute the relative height on.
    :return: The relative height between time, time-1 and time+1
    """
    if time < 1:
        raise ValueError("Relative height undefined for index less than 1")

    # Here we subtract 2 because -1 is the last element and we want the
    # specified time to have 1 element on each side in the list.
    elif time > len(smoothed_data)-2:
        raise ValueError("Relative height undefined for index at the end of "
                         "the time series.")
    t = time
    x = smoothed_data
    diff1 = abs(x[t] - x[t-1])
    diff2 = abs(x[t+1] - x[t])
    rel_height = max(diff1, diff2)
    return rel_height

def find_relative_heights(smoothed_data):
    relative_heights = []
    for i in range(1, len(smoothed_data)-1):
        next_rel_height = relative_height(smoothed_data, i)
        relative_heights.append(next_rel_height)
    return relative_heights

def find_time_indexes(relative_heights, peak_range):
    """
    :param time: the time offset
    :param points: the number of points used in moving average
    :return: the set of time indexes of the peaks in range h
    :l: lower bound of peak heights.
    :u: upper bound of peak heights.
    :r: number of channels
    :c: number of samples
    """
    l = peak_range[0]
    u = peak_range[1]
    rh = relative_heights
    time_indexes_in_range = []
    for i in range(0, len(relative_heights)):
        if l < rh[i] < u:
            # Add 1 because index of rel heights list is one less than the
            # index of the smoothed data.
            time_indexes_in_range.append(i+1)
    return time_indexes_in_range


def artifact_signals():
    pass


def maf_example():
    eeg_data = load_data(1, 'T')

    raw_signal = eeg_data[3][0][0]
    raw_signal = raw_signal[0:len(raw_signal)]
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