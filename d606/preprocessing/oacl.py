# This module contains implementation of ocular artifact detection and removal
from decimal import Decimal, getcontext
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from scipy.optimize import basinhopping
from scipy.optimize._basinhopping import Metropolis
from dataextractor import extract_trials_single_channel
from numpy import exp, array, cov, mean, inf, mat, copy, seterr, shape, frombuffer
from scipy.io import loadmat
from numpy.random import rand, uniform
from multiprocessing import Pool, Array, cpu_count
from ctypes import c_float, c_int
from itertools import product
from eval.timing import timed_block
from bisect import bisect_left


def _fixed_accept_reject(self, energy_new, energy_old):
    z = (energy_new - energy_old) * self.beta
    if z < -1:
        z = -1
    w = min(1.0, exp(-z))
    return w >= rand()

Metropolis.accept_reject = _fixed_accept_reject


def moving_avg_filter(chan_signal, m):
    # This function calculates the moving average filter of a single signal.
    # TODO: paper just uses symmetric MAF, but this loses information at ends!
    maf = []
    # We compute only so at each side of a point, we have (m-1)/2 elements.
    start = (m-1)/2
    end = len(chan_signal) - ((m-1) / 2)
    for i in range(start, end):
        ma_point = symmetric_moving_avg(chan_signal, m, i)
        maf.append(ma_point)
    return maf


def symmetric_moving_avg(chan_signal, m, t):
    """
    :param chan_signal: array of eeg time samples
    :param t: offset of sample to start with. Default is 0.
    :param m: number of neighboring points used in the filter.
    :return: smoothed signal.
    """
    n = len(chan_signal)
    if t + ((m - 1) / 2) >= n or t - (m / 2) < 0:
        raise ValueError("The desired range m for symmetric moving average "
                         "must be less for the given point t ")
    result = 0
    r = (m-1)/2
    for j in range(-r, r+1):
        val = chan_signal[t+j]
        result += val
    result *= 1 / float(m)
    return result


def find_relative_height(smooth_signal, time):
    """
    This function computes the relative height of points centered around
    point at param time.
    points
    :param smooth_signal: A list of (moving avg) points.
    :param time: the index in smoothed_data to compute the relative height on.
    :return: The relative height between time, time-1 and time+1
    """
    if time < 1:
        raise ValueError("Relative height undefined for index less than 1")

    # Here we subtract 2 because -1 is the last element and we want the
    # specified time to have 1 element on each side in the list.
    elif time > len(smooth_signal)-2:
        raise ValueError("Relative height undefined for index at the end of "
                         "the time series.")
    t = time
    x = smooth_signal
    diff1 = abs(x[t] - x[t-1])
    diff2 = abs(x[t+1] - x[t])
    rel_height = max(diff1, diff2)
    return rel_height


def zero_index(smooth_signal, i):
    a = smooth_signal[i]
    b = smooth_signal[i + 1]

    if b >= len(smooth_signal):
        # No polarity changes. So we just go with last element in the list.
        return True, i
    if a == 0.0:
        return True, i

    if a > 0 > b or a < 0 < b:
        m = min(a, b)
        if m == a:
            return True, i
        else:
            return True, i + 1
    else:
        return False, i


def find_relative_heights(smooth_signal):
    relative_heights = []
    zero_indexes = []
    for i in range(1, len(smooth_signal)-1):
        next_rel_height = find_relative_height(smooth_signal, i)
        zi = zero_index(smooth_signal, i)
        if zi[0] is True:
            zero_indexes.append(zi[1])
        relative_heights.append(next_rel_height)
    return relative_heights, zero_indexes


def find_peak_indexes(relative_heights, peak_range):
    l = peak_range[0]
    u = peak_range[1]
    rh = relative_heights

    # Add 1 because index of rel heights list is one less than the
    # index of the smoothed data.
    pt = [i+1 for i in range(0, len(rh)) if l < rh[i] < u]
    return pt


def find_artifact_ranges(smooth_signal, peak_indexes, zero_indexes):
    ranges = []
    latest_range = 0
    for peak in peak_indexes:
        if peak > latest_range:
            artifact_range = find_artifact_range(smooth_signal, peak, zero_indexes)
            if artifact_range[1] is None:
                latest_range = artifact_range[0]
                continue
            else:
                latest_range = artifact_range[1]
            ranges.append(artifact_range)
            residual_range = find_residuals(smooth_signal, artifact_range[1] + 5, zero_indexes)
            ranges.append(residual_range)
            latest_range = residual_range[1]
    return ranges


def find_residuals(smooth_signal, peak, zero_indexes):
    before, after = take_closest(zero_indexes, peak)
    return before, after


def take_closest(zero_indexes, peak):
    """
    Assumes myList is sorted. Returns closest values to myNumber.
    """
    pos = bisect_left(zero_indexes, peak)
    if pos == 0:
        return zero_indexes[0], zero_indexes[1]
    if pos == len(zero_indexes):
        return zero_indexes[-2], zero_indexes[-1]
    before = zero_indexes[pos - 1]
    after = zero_indexes[pos]

    return before, after


def find_artifact_range(signal_smooth, peak, zero_indexes):
    before, after = take_closest(zero_indexes, peak)
    #left = signal_smooth[0:peak]
    #left.reverse()
    #right = signal_smooth[peak+1:len(signal_smooth)]
    #before_i = nearest_zero_point(left, 0, 1)
    #after_i = nearest_zero_point(right, 0, 1)
    #before = (peak-1) - before_i
    #after = (peak+1) + after_i
    #if peak - before >= 62.6:
    #    return after + 25, None
    #if peak - before <= 25:
    #    return peak + 25 - peak - before, None
    #if after - peak >= 62.5:
    #    return after + 25, None
    #if after - peak <= 25:
    #    return after + 25, None
    if not 30 <= after - before <= 125:
        return after, None
    return before, after


def find_artifact_signal(peak_indexes, smooth_signal, zero_indexes):
    artifact_signal = [0.0 for x in range(0, len(smooth_signal))]
    ranges = sorted(find_artifact_ranges(smooth_signal, peak_indexes, zero_indexes), key=lambda z: z[0])
    for r in ranges:
        nzp_b = r[0]+1
        nzp_a = r[1]
        artifact_signal[nzp_b:nzp_a] = smooth_signal[nzp_b:nzp_a]
    return artifact_signal


def find_artifact_signals(raw_signal, m, range_list):
    artifact_signals = []
    smooth_signal = moving_avg_filter(raw_signal, m)
    rh, zi = find_relative_heights(smooth_signal)
    for range in range_list:
        peaks = find_peak_indexes(rh, range)
        artifact_signal = find_artifact_signal(peaks, smooth_signal, zi)
        artifact_signals.append(artifact_signal)
    return array(artifact_signals)


def nearest_zero_point(arr, a, b):
    if b >= len(arr):
        # No polarity changes. So we just go with last element in the list.
        return a
    x = arr[a]
    y = arr[b]
    if x == 0.0:
        return a

    elif is_cross_zero(x, y):
        m = min(abs(x), abs(y))
        min_index = (a if m == arr[a] else b)
        return min_index
    else:
        return nearest_zero_point(arr, b, b + 1)


def is_cross_zero(a, b):
    if a > 0 > b or a < 0 < b:
        return True
    else:
        return False


def covariance_matrix(artifact_signal, raw_signal):
    return cov(m=artifact_signal, y=raw_signal)


def correlation_vector(artifact_signals, signal):
    return artifact_signals * signal.transpose()


def column(matrix, i):
    return [row[i] for row in matrix]


def variance(vector):
    avg = int(mean(vector))
    return sum([pow(val - avg, 2) for val in vector]) / len(vector)


def objective_function(theta, b, labels, n_trials, trial_artifact_signals,
                       trial_signals):
    summa = 0
    theta_t = theta.transpose()

    y = labels
    z = []

    for i in range(n_trials):
        xa = mat(column(trial_artifact_signals, i))
        x0 = trial_signals[i]

        k1 = theta_t * (xa * xa.transpose()) * theta
        k2 = 2 * theta_t * (xa * x0.transpose())

        z.append([float(k1 - k2 + variance(x0.tolist()[0]) + b)])

    # TODO: find out if LR's C parameter should be optimized with Bayesian Optimization
    lr = LogisticRegression()
    lr.fit(z, labels)
    # Ignore OverflowWarnings that occur for exp(inf). The result is still correct.
    seterr(over='ignore')
    h = lr.predict_proba(z)
    seterr(over='warn')
    score = log_loss(y, h)

    return score

def remove_ocular_artifacts(raw_signal, theta, artifact_signals):
    A = theta.transpose().dot(artifact_signals).transpose()
    m = (len(raw_signal) - len(A)) / 2
    A = [0] * m + A.tolist() + [0] * m
    corrected_signal = (array(raw_signal) - array(A))
    return corrected_signal


def objective_function_aux(args, args2):
    arg1 = array([[args[k]] for k in xrange(len(args)-1)])
    arg2 = args[len(args)-1]
    return objective_function(arg1, arg2, *args2)


def extract_trials_array(signal, trials_start):
    n_trials = len(trials_start)
    concat_trials, concat_trials_start = extract_trials_single_channel(signal, trials_start)
    trials = [concat_trials[concat_trials_start[i]:concat_trials_start[i+1]]
              for i in xrange(n_trials-1)]
    trials += [concat_trials[concat_trials_start[n_trials-1]:]]
    return trials


class MyStepper:
    def __init__(self, stepsize=0.2):
        self.stepsize = stepsize

    def __call__(self, x):
        num_tethas = x.size - 1
        bounds = [[0, 1]] * num_tethas + [[-inf, 0]]
        s = self.stepsize
        while 1:
            x_old = copy(x)
            x[:num_tethas] += uniform(-s, s, shape(x[:num_tethas]))
            x[num_tethas] += uniform(-s * 10, s * 10, 1)
            test = [bounds[y][0] <= x[y] <= bounds[y][1] for y in range(0, len(bounds))]
            if all(test):
                break
            else:
                x = x_old
        return x


def get_theta(raw_signal, trials_start, labels, range_list, m):
    n_trials = len(trials_start)              # Number of trials
    artifact_signals = find_artifact_signals(raw_signal, m, range_list)
    trial_signals = mat(extract_trials_array(raw_signal, trials_start))
    trial_artifact_signals = [extract_trials_array(artifact_signals[i], trials_start)
                              for i in xrange(len(range_list))]
    mystepper = MyStepper()
    min_result = basinhopping(objective_function_aux,
                              [0.5] * (len(range_list)) + [0],
                              take_step=mystepper,
                              minimizer_kwargs={
                                  "bounds":[[0, 1]] * (len(range_list)) + [[-inf, 0]],
                                  "method":"SLSQP",
                                  "args":[labels, n_trials, trial_artifact_signals, trial_signals]
                              },
                              interval=7,
                              niter_success=25)
    filtering_param = array([[min_result.x[k]] for k in xrange(len(min_result.x) - 1)])
    # b = min_result.x[len(min_result.x) - 1]

    return filtering_param


def special_purpose_theta(args):
    index, entries, n_trials, n_labels, range_list, m = args
    run, channel = index
    start, end = entries
    print run, channel

    raw_signal = shared_array_oacl[start:end]
    artifact_signals = find_artifact_signals(raw_signal, m, range_list)
    trial_signals = mat(extract_trials_array(raw_signal, trials_start_oacl[run*n_trials:(run+1)*n_trials]))
    trial_artifact_signals = [extract_trials_array(artifact_signals[i],
                                                   trials_start_oacl[run*n_trials:(run+1)*n_trials])
                              for i in xrange(len(range_list))]
    mystepper = MyStepper()
    min_result = basinhopping(objective_function_aux,
                              [0.5] * (len(range_list)) + [0],
                              take_step=mystepper,
                              minimizer_kwargs={
                                  "bounds": [[0, 1]] * (len(range_list)) + [[-inf, 0]],
                                  "method": "SLSQP",
                                  "args": [labels_oacl[run*n_labels:(run+1)*n_labels], n_trials, trial_artifact_signals,
                                           trial_signals],
                              },
                              interval=7,
                              niter_success=25)
    filtering_param = array([[min_result.x[k]] for k in xrange(len(min_result.x) - 1)])
    # b = min_result.x[len(min_result.x) - 1]

    return index, filtering_param, artifact_signals


def init(shared_arr, trials_start, labels):
    global shared_array_oacl, trials_start_oacl, labels_oacl
    shared_array_oacl = shared_arr
    trials_start_oacl = trials_start
    labels_oacl = labels


def special_purpose_estimator(x, params):
    n_runs = len(x)
    n_channels = int(x[0][0].shape[0])
    len_chan = x[0][0].shape[1]
    run_len = n_channels * len_chan
    n_trials = len(x[0][1])
    n_labels = len(x[0][2])

    x_flat_size = sum([run[0].shape[0]*run[0].shape[1] for run in x])
    range_list, m, decimal_precision, trials = params
    getcontext().prec = decimal_precision

    shared_runs_base = Array(c_float, x_flat_size, lock=False)
    shared_trials_base = Array(c_int, n_trials*n_runs, lock=False)
    shared_labels_base = Array(c_int, n_labels*n_runs, lock=False)
    shared_runs_array = frombuffer(shared_runs_base, dtype=c_float)
    shared_trials_array = frombuffer(shared_trials_base, dtype=c_int)
    shared_labels_array = frombuffer(shared_labels_base, dtype=c_int)
    for i, z in enumerate(x):
        shared_runs_array[(i*n_channels*len_chan):((i+1)*n_channels*len_chan)] = z[0].flat
        shared_trials_array[i*n_trials:(i+1)*n_trials] = z[1][:]
        shared_labels_array[i*n_labels:(i+1)*n_labels] = z[2][:]

    pool = Pool(cpu_count(), initializer=init, initargs=(shared_runs_array, shared_trials_array, shared_labels_array))
    out = pool.map(special_purpose_theta, [(q, ((q[0]*run_len+q[1]*len_chan), (q[0]*run_len+len_chan*(q[1]+1))),
                                            n_trials, n_labels, range_list, m) for q in
                                           product(range(0, n_runs), range(0, n_channels))], chunksize=2)

    pool.close()
    pool.join()

    return out


def clean_signal(data, thetas, params):
    range_list, m, decimal_precision, _ = params
    getcontext().prec = decimal_precision
    channels, trials, labels, artifacts = data
    cleaned_signal = []
    for channel, theta in zip(channels, thetas):
        with timed_block('Cleaning signal '):
            artifacts_signals = find_artifact_signals(channel, m, range_list)
            cleaned_signal.append(remove_ocular_artifacts(channel, theta, artifacts_signals))
    return cleaned_signal


def clean_signal_multiproc(input_q, output_q, thetas, params):
    range_list, m, decimal_precision, trials = params
    getcontext().prec = decimal_precision
    if trials is True:
        data, trial_thetas, artifacts_signals, index = input_q.get()
    else:
        data, index = input_q.get()
    channels, trials, labels, artifacts = data
    cleaned_signal = []
    for i, (channel, theta) in enumerate(zip(channels, thetas)):
        print "Process " + str(index) + " is cleaning " + str(i)
        if trials is True:
            artifact_signal = artifacts_signals[i]
            theta = trial_thetas[i]
        else:
            artifact_signal = find_artifact_signals(channel, m, range_list)
        cleaned_signal.append(remove_ocular_artifacts(channel, theta, artifact_signal))
    if not output_q.full():
        output_q.put((cleaned_signal, index))
        output_q.close()
        output_q.join_thread()
    else:
        print "QUEUE IS FULL ????"
    print "Process " + str(index) + " exiting!!"


def estimate_theta(data, params):
    range_list, m, decimal_precision, _ = params
    getcontext().prec = decimal_precision
    channels, trials, labels, artifacts = data
    run_thetas = []
    for i, channel in enumerate(channels):
        print str(i)
        run_thetas.append(get_theta(channel, trials, labels, range_list, m))
    return run_thetas
