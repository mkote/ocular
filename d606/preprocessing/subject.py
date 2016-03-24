import scipy.io as sio


class Subject:
    path = "matfiles/"

    def __init__(self, sub_num, suffix):
        self._runs = []
        self._calibration = []
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

        for x in range(0, start):
            self.calibrations(mat["data"][0, x])

        for x in range(start, end):
            self.runs(mat["data"][0, x])

    def get_calibration(self):
        return self._calibration

    def set_calibration(self, val):
        self._calibration.append(val)

    def get_runs(self):
        return self._runs

    def set_runs(self, val):
        self._runs.append(val)

    runs = property(get_runs(), set_runs())
    calibrations = property(get_calibration(), set_calibration())
