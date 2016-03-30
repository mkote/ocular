import random

import math
import pywt
from matplotlib import pyplot
from pandas.algos import median
from pywt import _thresholding

NUM_SAMPLES = 64
MAX = pywt.swt_max_level(NUM_SAMPLES)


# http://iopscience.iop.org/article/10.1088/1741-2560/3/4/011/meta
# http://www.ee.ryerson.ca/~xzhang/publications/surespl.pdf
def softlike_threshold(v, t, k=3):
    for i in xrange(len(v)):
        y = v[i]
        if(y < -t):
            v[i] = y + t - t / (2*k+1)
        elif abs(y) <= t:
            v[i] = (1/((2*k+1)*pow(t, 2*k))) * pow(y, 2*k+1)
        elif y > t:
            v[i] = y - t + t / (2*k+1)
    return v

data = []
for i in range(0, NUM_SAMPLES):
    data += [random.randrange(-30, 30)]

# MAX = the number of times NUM_SAMPLES is divisible by 2
print(MAX)

for j in xrange(MAX):
    wt = pywt.swt(data, 'sym3', MAX)
    lvlj_approx = wt[j][0]
    lvlj_details = wt[j][1]

    sigma = median(lvlj_details) / 0.6745
    threshold = math.sqrt((2 * pow(sigma, 2)) * math.log(NUM_SAMPLES))
    lvlj_details = softlike_threshold(lvlj_details, threshold, 3)

    wt[j] = list(wt[j])
    wt[j][1] = lvlj_details

    #wt = _thresholding.less(wt, 1.7)
    iwt = pywt.iswt(wt, 'sym3')

    #print(data)
    #print(a)
    #print(b)
    #print(iwt)

    pyplot.axis([0, NUM_SAMPLES-1, -30, 30])
    pyplot.plot(data)
    pyplot.plot(iwt)
    pyplot.show()
