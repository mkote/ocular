import os
import subprocess
import time
import collections
import json
from sys import executable, exit
from main import main, translate_params, MainWrapper
from multiprocessing import freeze_support
from bayes_opt import BayesianOptimization

FIRST_SUBJECT = 1
LAST_SUBJECT = 9
CURRENT_SUBJECT = FIRST_SUBJECT
NUM_ITERATIONS = 200
resuming = False       # set to True if you are starting from existing work. DONT RESUME ON WORK FROM OTHER SUBJECTS.
STEP = 1 if FIRST_SUBJECT < LAST_SUBJECT else -1


def get_init_data():
    data = {}
    with open('../initdata.txt', 'rb') as fh:
        for line in fh:
            line_data = line.split(' ')
            data[line_data[0]] = {'band_range': line_data[1], 'n_comp': line_data[2]}

    return data


def new_optim_params():
    global resuming
    bounds = {'n_comp': (2, 22), 'band_range': (3, 8)}

    for subject in range(FIRST_SUBJECT, LAST_SUBJECT + STEP, STEP):
        num_iterations = NUM_ITERATIONS
        print("\n\nOptimizing hyperparameters for subject %i" % subject)
        wrapper = MainWrapper(subject)
        bo = BayesianOptimization(wrapper.main_aux, bounds)

        if resuming:
            init_knowledge = get_init_data()
            bo.initialize(init_knowledge)
            num_iterations = NUM_ITERATIONS - len(init_knowledge)
            resuming = False

        bo.maximize(acq='ei', n_iter=num_iterations)
        print('Result: %f %s' % (bo.res['max']['max_val'], bo.res['max']['max_params']))
        print(zip(bo.res['all']['values'], bo.res['all']['params']))


def optim_params():
    global resuming
    global CURRENT_SUBJECT
    for subject in range(FIRST_SUBJECT, LAST_SUBJECT + STEP, STEP):
        # if we are resuming, we expect some existing results. Check if the file exist.
        CURRENT_SUBJECT = subject
        expected_path = "braninpy/results_subject" + str(CURRENT_SUBJECT) + ".dat"
        file_exists = os.path.isfile(expected_path)
        if resuming and not file_exists:
            print "failed to resume work of subject " + str(CURRENT_SUBJECT) + " on file " + expected_path + " (file does not exist). Exiting..."
            exit(0)
        elif not resuming and file_exists:
            print "failed start new set of experiments. Result file already exist."
            exit(0)

        if not resuming:
            if os.path.isfile('braninpy/chooser.GPEIOptChooser.pkl'):
                os.remove('braninpy/chooser.GPEIOptChooser.pkl')
            if os.path.isfile('braninpy/chooser.GPEIOptChooser_hyperparameters.txt'):
                os.remove('braninpy/chooser.GPEIOptChooser_hyperparameters.txt')
            if os.path.isfile(expected_path):
                os.rename(expected_path, expected_path + '.' + str(time.time()))

        num_iter_done = 0;
        if os.path.isfile(expected_path):
            with open(expected_path) as f:
                num_iter_done = sum(1 for _ in f if _[0] != 'P')       # count number of lines in result file.
        if num_iter_done == 200:
            print "200 iterations already done for this subject. Exiting..."
            exit(0)
        
        for iteration in range(NUM_ITERATIONS if not resuming else 200-num_iter_done):        
            par = []
            condition = True
            while condition:
                old_path = os.getcwd()
                os.chdir('spearmintlite')
                subprocess.check_call([executable, 'spearmintlite.py', '--results', 'results_subject' + str(CURRENT_SUBJECT) + '.dat', '../braninpy'], cwd=os.getcwd())
                os.chdir(old_path)		
                params = get_params().rstrip()
                par = params.split(' ')	
                condition = any_params_out_of_bounds(par)

            n_comp, band_list, oacl_ranges, m = translate_params(par[2:])
            result, timestamp = main(n_comp, band_list, subject, oacl_ranges, m)
            insert_result(result, timestamp)
        
        # done with subject, set resuming to false.
        resuming = False
        # os.rename('braninpy/results.dat', 'braninpy/results' + str(subject) + '.dat')

def any_params_out_of_bounds(p):
    variables = json.load(open('braninpy/config.json'), object_pairs_hook=collections.OrderedDict).items()
    v = [var[1] for var in variables]
    are_params_out_of_bounds = any([float(p[2+i]) < v[i]['min'] or float(p[2+i]) > v[i]['max']
                                    for i in range(len(v))])
    return are_params_out_of_bounds
    
def get_params():
    with open('braninpy/results_subject' + str(CURRENT_SUBJECT) + '.dat', 'rb') as fh:
        last_line = ''
        for line in fh:
            last_line = line

    return last_line


def insert_result(result, time):
    file = "braninpy/results_subject" + str(CURRENT_SUBJECT) + ".dat"

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
    new_optim_params()
