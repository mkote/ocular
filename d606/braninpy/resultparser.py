

import collections

import json

import sys

from pprint import pprint

​

​

def get_all_params(path):

    result = []

    with open(path, 'rb') as fh:

        for line in fh:

            if line[0] != 'P':

                result.append(line.split(' '))

​

    return result

​

variables = json.load(open('./config.json'), object_pairs_hook=collections.OrderedDict).items()

variables = [v[1] for v in variables]

v = variables

​

path = './results.dat'

if len(sys.argv) > 1:

    if len(sys.argv[1]) > 1:

        path = sys.argv[1]

    else:

        path = './results' + str(sys.argv[1]) + '.dat'

​

params = get_all_params(path)

params_out_of_bounds = [p for p in params if any([float(p[2+i]) < v[i]['min'] or float(p[2+i]) > v[i]['max']

                                                  for i in range(len(v))])]

​

if len(params_out_of_bounds) > 0:

    print("Parameter(s) out of bounds: ")

    pprint(params_out_of_bounds)

​

errors = [float(p[0]) for p in params]

best_error = min(errors)

best_params = [p for p in params if float(p[0]) == best_error]

​

print(100 - best_error)

print(best_params)

