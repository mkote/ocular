import os
import subprocess
import time
from sys import executable
from main import main
from multiprocessing import freeze_support

SVC_KERNELS = ['linear', 'rbf', 'poly']
FIRST_SUBJECT = 1
LAST_SUBJECT = 9
NUM_ITERATIONS = 200


def optim_params():
    for subject in range(FIRST_SUBJECT, LAST_SUBJECT + FIRST_SUBJECT):
        if os.path.isfile('braninpy/chooser.GPEIOptChooser.pkl'):
            os.remove('braninpy/chooser.GPEIOptChooser.pkl')
        if os.path.isfile('braninpy/chooser.GPEIOptChooser_hyperparameters.txt'):
            os.remove('braninpy/chooser.GPEIOptChooser_hyperparameters.txt')
        if os.path.isfile('braninpy/results.dat'):
            os.rename('braninpy/results.dat', 'braninpy/results.dat.' + str(time.time()))

        for iteration in range(NUM_ITERATIONS):
            old_path = os.getcwd()
            os.chdir('spearmintlite')
            subprocess.check_call([executable, 'spearmintlite.py', '../braninpy'], cwd=os.getcwd())

            os.chdir(old_path)

            params = get_params()
            par = params.split(' ')
            n_comp = int(par[2])
            band_range = int(par[3])
            num_bands = int(36/band_range)
            band_list = [[4 + band_range * x, 4 + band_range * (x + 1)] for x in range(num_bands)]

            result, time = main(n_comp, band_list, subject)

            insert_result(result, timestamp)

        os.rename('braninpy/results.dat', 'braninpy/results' + str(subject) + '.dat')


def get_params():
    with open('braninpy/results.dat', 'rb') as fh:
        last_line = ''
        for line in fh:
            last_line = line

    return last_line


def insert_result(result, time):
    file = "braninpy/results.dat"

    # read the file into a list of lines
    lines = open(file, 'r').readlines()

    # now edit the last line of the list of lines
    last_line_list = lines[-1].rstrip().split(' ')
    last_line_list[0] = 100 - result
    last_line_list[1] = time
    print last_line_list
    last_line_string = ' '.join(map(str, last_line_list))
    lines[-1] = last_line_string + ' \n'
    print last_line_string
    # now write the modified list back out to the file
    open(file, 'w').writelines(lines)

if __name__ == '__main__':
    freeze_support()
    optim_params()
