#!/usr/bin/python
# seqproc_3Dplot.py - builds a CSV file necessary for the 3D plot


# imports
import subprocess


# destroy existing contents of output file
open('3dplot.csv', 'w').close()

# do for 5 windowsizes (3,5,7,9,11)
for windowsize in range(3, 12, 2):
    # do for 5 solvers (GaussianNB, SVC, MLP, RF, AdaBoost)
    for solver in range(5):
        # do for 10 train/test splits
        for split in range(10):
            print 'Processing for',windowsize,solver,split,'...'
            cmd = ['python','seqproc_setup.py', '-w', str(windowsize), '-e', str(solver), '-n']
            subprocess.Popen(cmd).wait()