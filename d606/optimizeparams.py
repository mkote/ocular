import os
import subprocess
from sys import executable
from time import sleep
from main import main, main_without_oacl
from multiprocessing import freeze_support

SVC_KERNELS = ['linear', 'rbf', 'poly']
FIRST_SUBJECT = 1
LAST_SUBJECT = 9
NUM_ITERATIONS = 200


def optim_params():
    for subject in range(FIRST_SUBJECT, LAST_SUBJECT + FIRST_SUBJECT):
        for iteration in range(NUM_ITERATIONS):
            old_path = os.getcwd()
            os.chdir('spearmintlite')
            p = subprocess.check_call([executable, 'spearmintlite.py', '../braninpy'], cwd=os.getcwd())

            os.chdir(old_path)

            params = get_params()
            par = params.split(' ')
            n_comp = int(par[2])
            band_range = int(par[3])
            num_bands = int(36/band_range)
            band_list = [[4 + band_range * x, 4 + band_range * (x + 1)] for x in range(num_bands)]
            s = int(par[4])
            r1 = int(par[5])
            r2 = int(par[6])
            space = int(par[7])
            m = int(par[8]) * 2 + 1
            oacl_ranges = ((s, s + r1), (space + s + r1 + 1, space + s + r1 + 1 + r2))

            result, time = main(n_comp, band_list, oacl_ranges, m, subject)

            insert_result(result, time)
        os.rename('braninpy/results.dat', 'braninpy/results' + str(subject) + '.dat')
        os.remove('braninpy/chooser.GPEIOptChooser.pkl')
        os.remove('braninpy/chooser.GPEIOptChooser_hyperparameters.txt')


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
