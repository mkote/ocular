from matplotlib import pyplot as plt
from matplotlib import style
from preprocessing.oacl import moving_avg_filter, find_relative_heights, find_peak_indexes, find_artifact_signal, zero_index
from preprocessing.dataextractor import load_data
from matplotlib2tikz import save as tikz_save
import numpy as np

def plot_example1():
    # load the data.
    eeg_data = load_data(1, 'T')
    sigfrom = 750*5
    sigto = sigfrom+750
    raw_signal = eeg_data[3][0][0][sigfrom:sigto]
    eog_signal1 = eeg_data[3][0][22][sigfrom:sigto]
    eog_signal2 = eeg_data[3][0][23][sigfrom:sigto]
    eog_signal3 = eeg_data[3][0][24][sigfrom:sigto]
    #labels = eeg_data[5][2]
    #n_trials = 48
    m = 11

    # DATA PREPROCESSING
    filtered = moving_avg_filter(raw_signal, m)
    padding = [0]*(m/2)
    padded_fsignal= list(padding)
    padded_fsignal.extend(filtered)
    padded_fsignal.extend(padding)
    rh, zero_indexes = find_relative_heights(filtered)
    ti = find_peak_indexes(rh, (4, 7))

    artifacts = [0]*(m/2)
    a = find_artifact_signal(ti, filtered, zero_indexes)
    artifacts.extend(a)
    #num_samples = len(raw)
    # prepare plot
    fig, ax = plt.subplots(2, sharex=True)
    #plt.subplot(211)
    # plot shit
    ax[0].plot([x for x in range(0, len(raw_signal))], raw_signal, 'b--', label='raw signal: x(t)')
    ax[0].plot([x for x in range(0, len(padded_fsignal))], padded_fsignal, 'g', label='smoothed signal: s(t)')
    ax[0].plot([x for x in range(0, len(artifacts))], artifacts, 'r', label='artifacts: a(t)')
    yax = [0]*len(zero_indexes)
    zero_indexesp = [x + m/2 for x in zero_indexes]
    ax[0].scatter(zero_indexesp, yax, color='red')
    ax[1].plot([x for x in range(0, len(eog_signal1))], eog_signal1, 'r', label='EOG 1')
    ax[1].plot([x for x in range(0, len(eog_signal2))], eog_signal2, 'b', label='EOG 2')
    ax[1].plot([x for x in range(0, len(eog_signal3))], eog_signal3, 'g', label='EOG 3')
    ax[0].legend(bbox_to_anchor=(0., 1.02, 1., .102),loc=4, prop={'size':8}, ncol=3,mode="expand",borderaxespad=0)
    ax[1].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=1, prop={'size':8}, ncol=3, mode="expand", borderaxespad=0)
    plt.ylabel('amplitude')
    plt.xlabel('time (t)')
    ax[0].axis([0, len(raw_signal), min(raw_signal), max(raw_signal)])
    ax[1].axis([0, len(raw_signal), min([min(eog_signal1), min(eog_signal2), min(eog_signal3)]),
                max([max(eog_signal1), max(eog_signal2), max(eog_signal3)])])
    plt.plot()
    plt.show() # showing the plot affects tikz save
    tikz_save('oacl-signals.tex',figureheight = '\\figureheight',figurewidth = '\\figurewidth')

def generate_zero_point_example():
    # load the data.
    eeg_data = load_data(1, 'T')
    raw_signal = eeg_data[5][0][4]
    raw_signal = raw_signal[1000:1020]
    labels = eeg_data[5][2]
    n_trials = 48
    m = 11

    # DATA PREPROCESSING
    raw = raw_signal
    filtered = moving_avg_filter(raw_signal, m)
    padded_fsignal= [0]*(m/2)
    padded_fsignal.extend(filtered)
    rh = find_relative_heights(filtered)
    ti = find_peak_indexes(rh, (3, 15))
    zero_indexes = []
    for x in xrange(len(filtered)-1):
        zero_indexes.append(zero_index(filtered, x))
    artifacts = [0]*(m/2)
    artifacts.extend(find_artifact_signal(ti, filtered, zero_indexes))
    print str(artifacts)
    num_samples = len(raw)
    # prepare plot
    fig, ax = plt.subplots()
    #plt.subplot(211)
    # plot shit
    ax.plot([x for x in range(0, len(raw_signal))], raw_signal, 'b--', label='raw signal: x(t)')
    ax.plot([x for x in range(0, len(padded_fsignal))], padded_fsignal, 'g', label='smoothed signal: s(t)')
    ax.plot([x for x in range(0, len(artifacts))], artifacts, 'r', label='artifacts: a(t)')
    ax.plot(zero_indexes, [0]*len(zero_indexes), 'ro', label='zero indexes')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102),loc=3,ncol=3,mode="expand",borderaxespad=0)
    plt.ylabel('amplitude')
    plt.xlabel('time (t)')
    #plt.axis([0, len(raw_signal), min(raw_signal), max(raw_signal)])
    plt.plot()
    plt.show()
    tikz_save('oacl-signals.tex',figureheight = '\\figureheight',figurewidth = '\\figurewidth')
plot_example1()