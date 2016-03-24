from d606.preprocessing.bandpass import *
import scipy.io as sio
from mne import create_info
import numpy as np


class Run:
    def __init__(self, argv):
        self._epochs = []
        self._events = []
        self._nr_epochs = 0
        self._freq = 0
        self.initialize_self(argv)

    def initialize_epochs(self, arr):
        epochs = FilterBank.initialize_banks(arr, n_banks=4,
                                             bank_range=[4, 40])
        extracted_epoch = []
        for k in np.arange(0, len(arr["trial"][0][0]), 1):
            start = arr["trial"][0][0][k][0]
            end = start + 1875
            for y in range(0, len(epochs)):
                extracted_epoch.append(epochs[y][0:22, start-1:end-1].copy())
            self.append(extracted_epoch.copy())
            extracted_epoch.clear()

    def initialize_events(self, arr):
        self.events = np.empty([self.nr_epochs, 3], dtype=int)
        temp_arr = np.empty([3])
        for k in np.arange(0, len(arr["trial"][0][0]), 1):
            temp_arr[0] = 0
            temp_arr[1] = (self.freq * 3)
            temp_arr[2] = arr["y"][0][0][k][0]
            self.set_row(k, temp_arr)

    def initialize_self(self, argv):
        self.freq = argv["fs"][0][0][0][0]
        self.initialize_epochs(argv)
        self.nr_epochs = len(self.epochs)
        self.initialize_events(argv)

    @property
    def nr_epochs(self):
        return self._nr_epochs

    @nr_epochs.setter
    def nr_epochs(self, nr):
        self._nr_epochs = nr

    @property
    def epochs(self):
        return self._epochs

    @epochs.setter
    def epochs(self, val):
        self._epochs = val

    def append(self, val):
        self.epochs.append(val)

    @property
    def freq(self):
        return self._freq

    @freq.setter
    def freq(self, val):
        self._freq = val

    @property
    def events(self):
        return self._events

    @events.setter
    def events(self, val):
        self._events = val

    def set_row(self, row_nr, val):
        self._events[row_nr] = val


class FilterBank:
    @classmethod
    def initialize_banks(cls, run, n_banks=2, bank_range=[8, 16]):
        filter_banks = []
        jump_size = bank_range[0] / n_banks
        for y in range(0, n_banks):
            start = bank_range[0] + y * jump_size
            end = start + jump_size
            filter_banks.append(bandpass_matrix(np.transpose(run["X"][0][0][:,
                                                0:22]),
                                                start, end))
        return filter_banks


class Subject:
    path = "matfiles/"

    def __init__(self, sub_num, suffix):
        self._runs = []
        self._file_name = sub_num
        self._full_file_name = None
        if suffix in ["T", "E"]:
            self._full_file_name = "A0" + sub_num + suffix + ".mat"
        else:
            self._full_file_name = "A0" + sub_num + "T" + ".mat"
        self.initialize_self()

    def initialize_self(self):
        mat = sio.loadmat(self.path + self._full_file_name)
        end = len(mat["data"][0])
        start = None
        for x in range(0, end):
            if mat["data"][0, x]["trial"].all().size > 0:
                start = x
                break
        for x in range(start, end):
            self.append(Run(mat["data"][0, x]))

    @property
    def runs(self):
        return self._runs

    @runs.setter
    def runs(self, val):
        self._runs = val

    def append(self, val):
        self._runs.append(val)

    def trials_with_label(self, val):
        if len(np.unique(val)) != 2:
            raise ValueError("The number of classes is not equal to 2")
        elif len(list(filter(lambda q: q not in [1, 2, 3, 4], np.unique(
                val).tolist()))) > 0:
            raise ValueError("Only values between 1 and 4 is accepted")
        else:
            epochs = []
            events = []
            for x in range(0, len(self.runs)):
                for y in range(0, len(self.runs[x].events)):
                    if self.runs[x].events[y][2] in val:
                        epochs.append(self.runs[x].epochs[y])
                        events.append(self.runs[x].events[y])
            return EegPackage(epochs, events)

    def trials_for_class_vs_rest(self, argv):
        classes = [1, 2, 3, 4]
        epochs = []
        events = []
        classes.remove(argv)
        for x in range(0, len(self.runs)):
            for y in range(0, len(self.runs[x].events)):
                epochs.append(self.runs[x].epochs[y].copy())
                events.append(self.runs[x].events[y].copy())
        for y in range(0, len(events)):
                if events[y][2] == argv:
                    pass
                else:
                    events[y][2] = classes[0]
        return EegPackage(epochs, events)


class EegPackage:
    def __init__(self, epochs, events):
        self._events = []
        self._epochs = []
        self._config = None
        self.initialize_self(epochs, events)

    def initialize_self(self, epochs, events):
        self.epochs = epochs
        self.events = np.asanyarray(events)
        self.config = create_info([str(x) for x in range(0, 22)], 250, "eeg")

    @property
    def epochs(self):
        return self._epochs

    @epochs.setter
    def epochs(self, val):
        self._epochs = val

    @property
    def events(self):
        return self._events

    @events.setter
    def events(self, val):
        self._events = val

    @property
    def config(self):
        return self._config

    @config.setter
    def config(self, val):
        self._config = val
