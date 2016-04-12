import real_main


def branin(n_comp, c):
    result = real_main.main(n_comp, c)

    print result
    return 100 - result


# Write a function like this called 'main'
def main(job_id, params):
    n_comp = params['n_comp'][0]
    c = params['X'][0]
    print 'Anything printed here will end up in the output directory for job #:', str(job_id)
    print params
    return branin(n_comp, c)
