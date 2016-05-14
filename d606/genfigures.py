from matplotlib import pyplot as plt
from matplotlib import style
from preprocessing.oacl import moving_avg_filter, find_relative_heights, find_peak_indexes, find_artifact_signal
from preprocessing.dataextractor import load_data
from matplotlib2tikz import save as tikz_save
import numpy as np

def plot_example():
	# load the data.
    eeg_data = load_data(1, 'T')
    raw_signal = eeg_data[5][0][4]
    raw_signal = raw_signal[1000:1500]
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
    artifacts = [0]*(m/2)
    artifacts.extend(find_artifact_signal(ti, filtered))
    num_samples = len(raw)
    # prepare plot
    fig, ax = plt.subplots()
    #plt.subplot(211)
    # plot shit
    ax.plot([x for x in range(0, len(raw_signal))], raw_signal, 'b--', label='raw signal: x(t)')
    ax.plot([x for x in range(0, len(padded_fsignal))], padded_fsignal, 'g', label='smoothed signal: s(t)')
    ax.plot([x for x in range(0, len(artifacts))], artifacts, 'r', label='artifacts: a(t)')
    plt.legend(bbox_to_anchor=(0., 1.02, 1., .102),loc=3,ncol=3,mode="expand",borderaxespad=0)
    plt.ylabel('amplitude')
    plt.xlabel('time (t)')
    #plt.axis([0, len(raw_signal), min(raw_signal), max(raw_signal)])
    plt.plot()
    #plt.show()
    tikz_save('oacl-signals.tex',figureheight = '\\figureheight',figurewidth = '\\figurewidth')

plot_example()