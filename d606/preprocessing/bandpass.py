from scipy.signal import butter, lfilter, filtfilt
from d606.preprocessing.dataextractor import *


def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    return b, a


def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order=order)
    y = lfilter(b, a, data)
    return y

def butter_bandpass_matrix(data, lowcut, highcut, fs, order=5):
    b, a = butter_bandpass(lowcut, highcut, fs, order)
    y = filtfilt(b, a, data, axis=1)
    return y

def bandpass_matrix(matrix, lowcut, highcut):
    """
    :param matrix: The matrix make bandpass on
    :param lowcut: the lowest frequency to use
    :param highcut: The highest frequency to use
    :return: a new matrix which have been bandpass filtered
    """
    fs = 250
    new_matrix = butter_bandpass_matrix(matrix, lowcut, highcut, fs, order=6)
    # height = matrix.shape[0]
    # new_matrix = np.empty((0, matrix.shape[1]))
    #
    # # Not sure about the order in the call to b_b_f, might need adjustment
    # for x in range(0, height):
    #     row = matrix[x, :]
    #     new_row = butter_bandpass_filter(row, lowcut, highcut, fs, order=6)
    #     new_matrix = np.append(new_matrix, [new_row], axis=0)

    return new_matrix
