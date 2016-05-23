from matplotlib import pyplot as plt
from matplotlib import style
from preprocessing.oacl import moving_avg_filter, find_relative_heights, find_peak_indexes, find_artifact_signal, zero_index
from preprocessing.dataextractor import load_data
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
    fig, ax = plt.subplots(2, sharex=True, figsize=(10,5))
    #plt.subplot(211)
    # plot shit
    ax[0].plot([x for x in range(0, len(raw_signal))], raw_signal, 'b', label='raw signal: x(t)')
    ax[0].plot([x for x in range(0, len(padded_fsignal))], padded_fsignal, 'g', label='smoothed signal: s(t)')
    ax[0].plot([x for x in range(0, len(artifacts))], artifacts, 'r', label='artifacts: a(t)')
    yax = [0]*len(zero_indexes)
    zero_indexesp = [x + m/2 for x in zero_indexes]
    ax[0].scatter(zero_indexesp, yax, color='red')
    ax[1].plot([x for x in range(0, len(eog_signal1))], eog_signal1, 'r', label='EOG 1')
    ax[1].plot([x for x in range(0, len(eog_signal2))], eog_signal2, 'b', label='EOG 2')
    ax[1].plot([x for x in range(0, len(eog_signal3))], eog_signal3, 'g', label='EOG 3')
    ax[0].legend(bbox_to_anchor=(0., 1.02, 1., .102),loc=4, prop={'size':11}, ncol=3,mode="expand",borderaxespad=0)
    ax[1].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=4, prop={'size':11}, ncol=3, mode="expand", borderaxespad=0)
    plt.ylabel('amplitude')
    plt.xlabel('time (t)')
    ax[0].axis([0, len(raw_signal), min(raw_signal), max(raw_signal)])
    ax[1].axis([0, len(raw_signal), min([min(eog_signal1), min(eog_signal2), min(eog_signal3)]),
                max([max(eog_signal1), max(eog_signal2), max(eog_signal3)])])
    plt.plot()
    #plt.show() # showing the plot affects tikz save
    plt.tight_layout(pad=2.0, w_pad=0.5, h_pad=2.5)
    plt.savefig('oacl-signals.png', format='png', dpi=300)

plot_example1()
