from collections import namedtuple
from multiprocessing import Process, Queue
from d606.preprocessing.oacl import clean_eeg
import numpy as np


def remake_trial(raw_data):
    trial = namedtuple('EEG', ['matrix', 'trials', 'labels', 'artifacts'])
    temp_trial = []
    temp_part_run = []
    processes = []
    result = []
    first_index = (raw_data.index(x) for x in raw_data if len(x[1]) > 0).next()
    input_q = Queue(len(raw_data) - first_index)
    output_q = Queue(len(raw_data) - first_index)
    for x in range(0, first_index):
        temp_trial.append(raw_data[x][0])
        temp_trial.append(raw_data[x][1])
        temp_trial.append(raw_data[x][2])
        temp_trial.append(raw_data[x][3])
        result.append(trial(*temp_trial))
        temp_trial = []
    for x in range(first_index, len(raw_data)):
        input_q.put((raw_data[x], x))
        temp_trial.append([])
        temp_trial.append(raw_data[x][1])
        temp_trial.append(raw_data[x][2])
        temp_trial.append(raw_data[x][3])
        temp_part_run.append(temp_trial)
        temp_trial = []
    for x in range(first_index, len(raw_data)):
        p = Process(target=clean_eeg, args=(input_q, output_q,
                                            ((4, 6), (8, 15)), 11))
        p.start()
        processes.append(p)

    temp = []
    for x in range(0, 6):
        temp.append(output_q.get())

    for p in processes:
        p.join()

    for x in range(0, len(temp_part_run)):
        temp_part_run[temp[x][1]-3][0].extend(np.array(temp[x][0]))

    for x in range(0, len(temp_part_run)):
        result.append(trial(*temp_part_run[x]))

    return result
