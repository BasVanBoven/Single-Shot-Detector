#!/usr/bin/python
# seqproc_3Dplot.py - builds a CSV file necessary for the 3D plot


# imports
import subprocess
import argparse
import numpy as np


# handle input arguments
parser = argparse.ArgumentParser(description='Generate data for producing a 3D plot of the sequence processor variables')
parser.add_argument('-d', '--destroy', default=False, action='store_true', help='destroy existing contents of output file')
args = parser.parse_args()


# destroy existing file if necessary
if args.destroy:
    open('3dplot.csv', 'w').close()

# do for 5 windowsizes (3,5,7,9,11)
for windowsize in range(3, 12, 2):
    # do for 5 solvers (GaussianNB, SVC, MLP, AdaBoost, RF)
    for solver in range(5):
        # do for 1 readymade train/test split
        for split in range(1):
            print 'Processing for',windowsize,solver,split,'...'
            cmd = ['python','seqproc_setup.py', '-w', str(windowsize), '-e', str(solver), '-n', '-l']
            subprocess.Popen(cmd).wait()

# translate generated file to a plottable array
source = np.genfromtxt('3dplot.csv', delimiter=',', dtype=float)
plottable = np.zeros((5, 5), dtype=float)
for row in source:
    plottable[(int(row[0])/2)-1,int(row[1])] += row[2]
# divide by the number of train/test splits (currently 1)
plottable /= 1

# print output
print plottable