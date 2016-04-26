import ast
import os

import real_main
from eval.timing import timed_block


def branin(n_comp, c, kernel, band_list, oacl_ranges, m):

    old_path = os.getcwd()
    print old_path
    if 'd606' not in old_path:
        os.chdir('../../../d606')

    with timed_block('Iteration '):
        result = real_main.main(n_comp, c, kernel, band_list, oacl_ranges, m)

    os.chdir(old_path)

    print result
    return 100 - result


# Write a function like this called 'main'
def main(job_id, params):
    n_comp = params['n_comp'][0]
    c = params['C'][0]
    kernel = str(params['kernel'][0])
    band_list = ast.literal_eval(params['band_list'][0])
    r1 = params['r1'][0]
    r2 = params['r2'][0]
    r3 = params['r3'][0]
    r4 = params['r4'][0]
    oacl_ranges = ((r1, r2), (r3, r4))
    m = params['m'][0]
    print 'Anything printed here will end up in the output directory for job #:', str(job_id)
    print params
    return branin(n_comp, c, kernel, band_list, oacl_ranges, m)
