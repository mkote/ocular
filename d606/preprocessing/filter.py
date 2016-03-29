from d606.preprocessing.bandpass import bandpass_matrix


class Filter:
    def __init__(self, range_spec):
        self._range = []
        self.band_range = range_spec

    def filter(self, data):
        filters = []
        if len(self.band_range) <= 0:
            ValueError("You dumb fuck, you need to specify a legal range")
        else:
            for x in range(0, len(self.band_range)):
                filters.append(bandpass_matrix(data, self.band_range[x][0],
                                               self.band_range[x][1]))
            return filters

    def get_range(self):
        return self._range

    def set_range(self, val):
        self._range = val

    band_range = property(get_range, set_range)