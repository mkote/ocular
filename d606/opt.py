import os
import subprocess
import collections
import json
from sys import executable
from approach import func

NUM_ITERATIONS = 200


def optim_params():
    num_iter_done = 0;
    for iteration in range(NUM_ITERATIONS):
        par = []
        condition = True
        while condition:
            old_path = os.getcwd()
            os.chdir('spearmintlite')
            subprocess.check_call(
                [executable, 'spearmintlite.py', '--results', 'results_subject' + str(1) + '.dat',
                 '../braninpy'], cwd=os.getcwd())
            os.chdir(old_path)
            params = get_params().rstrip()
            par = params.split(' ')
            condition = any_params_out_of_bounds(par)

            oacl_range, k, m = translate_params(par[2:])
            result, timestamp = func(oacl_range, k, m)
            insert_result(result, timestamp)


def translate_params(par):
    oacl_range = (int(par[0]), int(par[0])+int(par[1]))
    m = int(par[2])*2
    if m % 2 == 0:
        m += 1
    k = float(par[3])
    return oacl_range, k, m

def any_params_out_of_bounds(p):
    variables = json.load(open('braninpy/config.json'), object_pairs_hook=collections.OrderedDict).items()
    v = [var[1] for var in variables]
    are_params_out_of_bounds = any([float(p[2 + i]) < v[i]['min'] or float(p[2 + i]) > v[i]['max']
                                    for i in range(len(v))])
    return are_params_out_of_bounds


def get_params():
    with open('braninpy/results_subject' + str(1) + '.dat', 'rb') as fh:
        last_line = ''
        for line in fh:
            last_line = line

    return last_line


def insert_result(result, time):
    file = "braninpy/results_subject" + str(1) + ".dat"

    # read the file into a list of lines
    lines = open(file, 'r').readlines()

    # now edit the last line of the list of lines
    last_line_list = lines[-1].rstrip().split(' ')
    last_line_list[0] = result
    last_line_list[1] = time
    print last_line_list
    last_line_string = ' '.join(map(str, last_line_list))
    lines[-1] = last_line_string + ' \n'
    print last_line_string
    # now write the modified list back out to the file
    open(file, 'w').writelines(lines)


if __name__ == '__main__':
    optim_params()
