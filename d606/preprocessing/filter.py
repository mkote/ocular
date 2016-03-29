from d606.preprocessing.bandpass import bandpass_matrix


class Filter:
    def __init__(self, range_spec):
        self._range = []
        self.range = range_spec

    def filter(self, data):
        filters = []
        if len(self.range) <= 0:
            ValueError("You dumb fuck, you need to specify a legal range")
        else:
            for x in range(0, len(self.range)):
                filters.append(bandpass_matrix(data, self.range[0][0],
                                               self.range[0][1]))
            return filters

    def get_range(self):
        return self._range

    def set_range(self, val):
        self._range = val

    range = property(get_range, set_range)
