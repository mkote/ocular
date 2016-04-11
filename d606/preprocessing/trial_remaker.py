from collections import namedtuple
from d606.preprocessing.oaclbase import OACL
from numpy import array


def remake_trial(raw_data, arg_oacl=None):
    trial = namedtuple('EEG', ['matrix', 'trials', 'labels', 'artifacts'])
    temp_trial = []
    result = []

    if arg_oacl is None:
        oacl = OACL(multi_run=True, ranges=((3, 7),(7, 15)))
    else:
        oacl = arg_oacl

    first_index = (raw_data.index(x) for x in raw_data if len(x[1]) > 0).next()

    for x in range(0, first_index):
        temp_trial.append(raw_data[x][0])
        temp_trial.append(raw_data[x][1])
        temp_trial.append(raw_data[x][2])
        temp_trial.append(raw_data[x][3])
        result.append(trial(*temp_trial))
        temp_trial = []

    if arg_oacl is None:
        cleaned_signal = oacl.fit_transform(raw_data[first_index:], raw_data[first_index:][2])

    else:
        cleaned_signal = oacl.transform(raw_data[first_index:])

    for x in range(first_index, len(raw_data)):
        temp_trial.append(array(cleaned_signal[x - first_index]))
        temp_trial.append(raw_data[x][1])
        temp_trial.append(raw_data[x][2])
        temp_trial.append(raw_data[x][3])
        result.append(trial(*temp_trial))
        temp_trial = []

    return result, oacl


def remake_single_trial(raw_data):
    trial = namedtuple('EEG', ['matrix', 'trials', 'labels', 'artifacts'])
    temp_trial = []
    result = []

    oacl = OACL()

    cleaned_signal = oacl.fit_transform(raw_data, raw_data[2])


    temp_trial.append(cleaned_signal)
    temp_trial.append(raw_data[1])
    temp_trial.append(raw_data[2])
    temp_trial.append(raw_data[3])
    result.append(trial(*temp_trial))
    temp_trial = []

    return result, oacl