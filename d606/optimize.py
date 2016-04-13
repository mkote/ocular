import real_main
import ast
import os


def branin(n_comp, c, kernel, band_list, oacl_ranges):
    result = real_main.main(n_comp, c, kernel, band_list, oacl_ranges)

    print result
    return 100 - result


# Write a function like this called 'main'
def main(job_id, params):
    n_comp = params['n_comp'][0]
    c = params['C'][0]
    kernel = str(params['kernel'][0])
    band_list = ast.literal_eval(params['band_list'][0])
    oacl_ranges = ast.literal_eval(params['oacl_ranges'][0])
    print 'Anything printed here will end up in the output directory for job #:', str(job_id)
    print params
    return branin(n_comp, c, kernel, band_list, oacl_ranges)

os.system('cd ../lib/spearmint/bin && '
          './spearmint ../../../d606/config.pb --driver=local '
          '--method=GPEIOptChooser -w --method-args=noiseless=1')
