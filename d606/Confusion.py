from preprocessing.dataextractor import load_data
from preprocessing.oacl import moving_avg_filter, find_relative_heights, find_peak_indexes, find_artifact_signal
import math


def tricon(g, h, places):
    la = (g, h)
    return True if la in places else False


def artcon(art):
    summa = []
    test = [0.0 for l in range(750)]
    for t in art:
        summa.append(sum([math.fabs(t[x]) for x in range(len(t))]))
    return True if sum(summa) > 0 else False


def find_art(channels, r, m):
    art = []
    for chn in channels:
        filtered = moving_avg_filter(chn, m)
        padding = [0.0] * (m / 2)
        padded_fsignal = list(padding)
        padded_fsignal.extend(filtered)
        padded_fsignal.extend(padding)
        rh, zero_indexes = find_relative_heights(filtered)
        ti = find_peak_indexes(rh, r)

        artifacts = list(padding)
        a = find_artifact_signal(ti, filtered, zero_indexes)
        artifacts.extend(a)
        artifacts.extend(padding)
        art.append(artifacts)
    return art


eeg_data = load_data(9, 'T')
m = 17
oacl_range = (5, 22)
first_index = (eeg_data.index(x) for x in eeg_data if len(x[1]) > 0).next()
places = []
for i, x in enumerate(eeg_data[first_index:]):
    for j, y in enumerate(x[3]):
        if y == 1:
            places.append((i, j))
r1c1 = 0
r2c1 = 0
r1c2 = 0
r2c2 = 0
for z, x in enumerate(eeg_data[first_index:]):
    trial_contains = False
    artifact_contains = False
    for w, y in enumerate(x[1]):
        sigfrom = y+700
        sigto = sigfrom+750
        chns = [x[0][k][sigfrom:sigto] for k in range(22)]
        art = find_art(chns, oacl_range, m)
        trial_contains = tricon(z, w, places)
        artifact_contains = artcon(art)
        if trial_contains is True:
            if artifact_contains is True:
                r1c1 += 1
            else:
                r1c2 += 1
        else:
            if artifact_contains is True:
                r2c1 += 1
            else:
                r2c2 += 1

print "Trial marked and artifact found: " + str(r1c1)
print "Trial marked and no artifact found: " + str(r1c2)
print "Trial not marked and artifact found: " + str(r2c1)
print "Trial not marked and no artifact found: " + str(r2c2)
