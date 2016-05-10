import sys


def get_all_params(path):
    result = []
    with open(path, 'rb') as fh:
        for line in fh:
            if line[0] != 'P':
                result.append(line.split(' '))

    return result

path = './results.dat'
if len(sys.argv) > 1:
    if len(sys.argv[1]) > 1:
        path = sys.argv[1]
    else:
        path = './results' + str(sys.argv[1]) + '.dat'

params = get_all_params(path)
errors = [float(p[0]) for p in params]
best_error = min(errors)
best_params = [p for p in params if float(p[0]) == best_error]

print(100 - best_error)
print(best_params)
