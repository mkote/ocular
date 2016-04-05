from d606.preprocessing.bandpass import bandpass_matrix
from scipy.signal import iirfilter


class Filter:
    def __init__(self, range_spec):
        self._range = []
        self.band_range = range_spec
        # self.filters = []
        # self.create_filters()

    def create_filters(self):
        nyq = 0.5 * 250
        new_range = [(x / nyq, y / nyq) for x, y in self.band_range]

        self.filters = [iirfilter(6, x, rp=3, rs=50, btype='band',
                                  analog=True, ftype='ellip')
                        for x in new_range]

    def filter(self, data):
        filters = []
        if len(self.band_range) <= 0:
            ValueError("You need to specify a legal range")
        else:
            for x in range(0, len(self.band_range)):
                filters.append(bandpass_matrix(data, self.band_range[x][0],
                                               self.band_range[x][1]))
                # filters.append(filtfilt(self.filters[x][0], self.filters[x][
                #  1], data))
            return filters

    def get_range(self):
        return self._range

    def set_range(self, val):
        self._range = val

    band_range = property(get_range, set_range)
