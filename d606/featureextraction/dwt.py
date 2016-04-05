from pywt import dwt
from d606.preprocessing.dataextractor import *

jumplist = [0, 1, 2]


def dwt1(mat_data):
    for i, run in enumerate(mat_data):
        if i in jumplist:
            pass

        a, b, c, d = run

        row = a[0, :]
        new_row = row[0:3600]
        ca, cd = dwt(new_row, 'db1')
        ca, cd = dwt(ca, 'db1')
        print(ca.shape)
