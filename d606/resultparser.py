import collections
import json
import os
import sys
from pprint import pprint


def get_all_params(path):
    result = []
    with open(path, 'rb') as fh:
        for line in fh:
            if line[0] != 'P':
                result.append(line.rstrip().split(' '))

    return result

def best_subject_params(subject):
    if len(sys.argv) < 2:
        print("Not enough arguments: enter a number corresponding to a subject")
        sys.exit(0)
    path = './braninpy/results_subject' + str(subject) + '.dat'

    params = get_all_params(path)

    if os.path.isfile('./braninpy/config.json'):
        variables = json.load(open('./braninpy/config.json'), object_pairs_hook=collections.OrderedDict).items()
        v = [var[1] for var in variables]
        params_out_of_bounds = [p for p in params if any([float(p[2+i]) < v[i]['min'] or float(p[2+i]) > v[i]['max']
                                                          for i in range(len(v))])]

        if len(params_out_of_bounds) > 0:
            print("Parameter(s) out of bounds (" + str(len(params_out_of_bounds)) + "):")
            pprint(params_out_of_bounds)

    print("Unique parameter combinations: " + str(len(set([';'.join(p[2:]) for p in params]))) + " out of " + str(len(
        params)) + " iterations.")

    errors = [float(p[0]) for p in params]
    best_error = min(errors)
    best_params = [p for p in params if float(p[0]) == best_error]

    return best_error, best_params

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Not enough arguments: enter a number corresponding to a subject")
        sys.exit(0)
    best_error, best_params = best_subject_params(sys.argv[1])
    print("Best accuracy: " + str(100 - best_error))
    print("Best parameters: " + str(set([' '.join(p[2:]).rstrip() for p in best_params])))