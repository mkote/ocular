from main import main
import os
import subprocess
import time
from sys import executable
from main import main
from multiprocessing import freeze_support

subject = 1
params = "P P 12 4 10 1 1 2 8"
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
result, timestamp = main(n_comp, band_list, subject, oacl_ranges, m)

print "Accuracy: " + str(result)
print "Time: " + str(timestamp)