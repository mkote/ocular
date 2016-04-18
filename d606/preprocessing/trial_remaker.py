from collections import namedtuple
from preprocessing.oaclbase import OACL
from numpy import array
import preprocessing.searchgrid as search

# Variable must match name of type to be pickleable.
EEG = namedtuple('EEG', ['matrix', 'trials', 'labels', 'artifacts'])
trial = EEG


def remake_trial(raw_data, arg_oacl=None):
    result = []

    if arg_oacl is None:
        m = search.grid.m if 'm' in search.grid._fields else 11
        if m % 2 == 0:
            m += 1
        ranges = search.grid.oacl_ranges if 'oacl_ranges' in search.grid._fields else ((3, 7), (7, 15))
        oacl = OACL(multi_run=True, ranges=ranges, trials=True, m=m)
    else:
        oacl = arg_oacl
        oacl.trials = False

    first_index = (raw_data.index(x) for x in raw_data if len(x[1]) > 0).next()

    for raw_trial in raw_data[:first_index]:
        result.append(trial(*raw_trial[:4]))

    if arg_oacl is None:
        cleaned_signal = oacl.fit_transform(raw_data[first_index:], raw_data[first_index:][2])

    else:
        cleaned_signal = oacl.transform(raw_data[first_index:])

    for x, raw_trial in enumerate(raw_data[first_index:]):
        first_cleaned = array(cleaned_signal[x])
        rest = raw_trial[1:4]
        result.append(trial(first_cleaned, *rest))

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
