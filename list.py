#!/usr/bin/python
# list.py - lists number of frames (files) per folder


# imports
import os
import sys


# perform all tests in testsets
for root, dirs, files in os.walk(os.getcwd()):
    for dir in dirs:
        num = len([name for name in os.listdir(dir) if os.path.isfile(os.path.join(dir, name))])
        print dir + ';' + str(num)