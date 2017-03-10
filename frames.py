#!/usr/bin/python
# frames.py - batch extracts frames from videos


# imports
import os
import sys
import argparse


# handle input arguments
parser = argparse.ArgumentParser(description='Train a Single Shot Detector.')
parser.add_argument('target_dir', help='directory which is to be walked recursively')
parser.add_argument('-l', '--list', default=False, action='store_true', help='list the available video files instead of converting them')
args = parser.parse_args()


# perform all tests in testsets
for root, dirs, files in os.walk(args.target_dir):
    for name in files:
        name, ext = os.path.splitext(name)
        if args.list == True:
            if (ext.lower().endswith(('.mp4', '.avi', '.mov'))):
                # print csv
                print (root+' '+name+ext)
        else:
            if (ext.lower().endswith(('.mp4', '.avi', '.mov'))):
                # create directory if it does not exist
                target_dir = 'frames'+'/'+name
                if not os.path.exists(target_dir):
                    os.makedirs(target_dir)
                # call ffmpeg
                os.system('ffmpeg -i "'+root+'/'+name+ext+'" -r 1 "'+'frames'+'/'+name+'"/"'+name+'"_%4d.jpg')