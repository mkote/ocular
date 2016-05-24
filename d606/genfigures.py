from matplotlib import pyplot as plt
from matplotlib import style
from preprocessing.oacl import moving_avg_filter, find_relative_heights, find_peak_indexes, find_artifact_signal, zero_index
from preprocessing.dataextractor import load_data
import numpy as np

def plot_example1():
    # load the data.
    eeg_data = load_data(1, 'T')
    places = []
    for i, x in enumerate(eeg_data[3:]):
        for j, y in enumerate(x[3]):
            if y == 1:
                places.append((i, j))
    org_ranges = ((3.2, 7),)
    for i, j in places:
        run = i+3
        sigfrom = eeg_data[run][1][j]+700
        sigto = sigfrom+750
        chns = [eeg_data[run][0][0][sigfrom:sigto], eeg_data[run][0][1][sigfrom:sigto], eeg_data[run][0][2][sigfrom:sigto],
                eeg_data[run][0][3][sigfrom:sigto], eeg_data[run][0][3][sigfrom:sigto], eeg_data[run][0][3][sigfrom:sigto]]
        eog_signal1 = [0.0] * 3 + moving_avg_filter(eeg_data[run][0][22][sigfrom:sigto], 7) + [0.0] * 3
        eog_signal2 = [0.0] * 3 + moving_avg_filter(eeg_data[run][0][23][sigfrom:sigto], 7) + [0.0] * 3
        eog_signal3 = [0.0] * 3 + moving_avg_filter(eeg_data[run][0][24][sigfrom:sigto], 7) + [0.0] * 3
        #labels = eeg_data[5][2]
        #n_trials = 48
        m = 13

        # DATA PREPROCESSING
        art = []
        smt = []

        for k, e in enumerate(chns):
            if k > 0:
                layer = 1
            filtered = moving_avg_filter(e, m)
            padding = [0]*(m/2)
            padded_fsignal= list(padding)
            padded_fsignal.extend(filtered)
            padded_fsignal.extend(padding)
            smt.append(padded_fsignal)
            rh, zero_indexes = find_relative_heights(filtered)
            r1, r2 = org_ranges[0]
            r1 = r1 - k*0.08
            ti = find_peak_indexes(rh, (r1, r2))

            artifacts = list(padding)
            a = find_artifact_signal(ti, filtered, zero_indexes)
            artifacts.extend(a)
            artifacts.extend(padding)
            art.append(artifacts)
        #num_samples = len(raw)
        # prepare plot
        fig, ax = plt.subplots(len(chns)+1, sharex=True, figsize=(10,5))
        #plt.subplot(211)
        # plot shit
        for i, (e, f, g) in enumerate(zip(chns, smt, art)):
            ax[i].plot([x for x in range(0, len(e))], e, 'b', label='raw signal: x'+str(i)+'(t)')
            ax[i].plot([x for x in range(0, len(f))], f, 'g', label='smoothed signal: s'+str(i)+'(t)')
            ax[i].plot([x for x in range(0, len(g))], g, 'r', label='artifacts: a'+str(i)+'(t)')
        # yax = [0]*len(zero_indexes)
        # zero_indexesp = [x + m/2 for x in zero_indexes]
        # ax[0].scatter(zero_indexesp, yax, color='red')
        ax[len(chns)].plot([x for x in range(0, len(eog_signal1))], eog_signal1, 'r', label='EOG 1')
        ax[len(chns)].plot([x for x in range(0, len(eog_signal2))], eog_signal2, 'b', label='EOG 2')
        ax[len(chns)].plot([x for x in range(0, len(eog_signal3))], eog_signal3, 'g', label='EOG 3')
        # ax[0].legend(bbox_to_anchor=(0., 1.02, 1., .102),loc=4, prop={'size':11}, ncol=3,mode="expand",borderaxespad=0)
        # ax[len(chns)].legend(bbox_to_anchor=(0., 1.02, 1., .102), loc=4, prop={'size':11}, ncol=3, mode="expand", borderaxespad=0)
        plt.ylabel('amplitude')
        # plt.xlabel('time (t)')

        plt.plot()
        plt.show() # showing the plot affects tikz save
        # plt.tight_layout(pad=2.0, w_pad=0.5, h_pad=2.5)
        # plt.savefig('oacl-signal9'+str(run)+str(j)+'.png', format='png', dpi=300)
        plt.close()

plot_example1()
