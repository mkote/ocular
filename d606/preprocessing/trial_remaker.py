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


def remake_single_run_transform(raw_data, arg_oacl=None):
    trial = namedtuple('EEG', ['matrix', 'trials', 'labels', 'artifacts'])
    result = []
    oacl = ''

    if arg_oacl is None:
        oacl = OACL()
    else:
        oacl = arg_oacl

    cleaned_signal = oacl.transform(raw_data)

    for i, raw_trial in enumerate(raw_data):
        rest = raw_trial[1:4]
        result.append(trial(array(cleaned_signal[i]), *rest))

    return result
