#!/usr/bin/python
# video.py - prepares video's for training the Sequence Processor

# input: a (set of) folders containing videoname.ext and videoname.txt in the same directory
# output: one folder per video, containing jpg-frames (both unannotated and annotated), json-ssd (from DeepDetect) and csv

# output folder structure:
# video_output/VIDEONAME/jpg_unannotated/VIDEONAME_FRAMENUMBER.jpg
# video_output/VIDEONAME/jpg_annotated/VIDEONAME_FRAMENUMBER.jpg
# video_output/VIDEONAME/json_ssd/VIDEONAME_FRAMENUMBER.json
# video_output/VIDEONAME/VIDEONAME.csv

# prerequisites:
# sudo apt-get install software-properties-common
# sudo add-apt-repository ppa:mc3man/trusty-media
# sudo apt-get update
# sudo apt-get install ffmpeg
# sudo pip install ffmpy


# imports
import argparse
import math
import os
import shutil
import stat
import subprocess
import sys
import random
import PIL
import time
import datetime
import random
import caffe
import cv2
import ffmpy
import re
from google.protobuf import text_format
from PIL import Image
from dd_client import DD


# handle input arguments
parser = argparse.ArgumentParser(description='Process input data for training a Sequence Processor.')
parser.add_argument('builddir', help='build (timestamp only) that is to be tested')
parser.add_argument('-i', '--iter', type=int, default=0, help='use a specific model iteration')
parser.add_argument('-f', '--framerate', type=float, default=1.0, help='how many frames to store and process per second')
parser.add_argument('-c', '--confidence-threshold', type=float, default=0.1, help='keep detections with confidence above threshold')
args = parser.parse_args()


# global variables
folder_input = os.path.join(os.getcwd(), 'video', 'input')
folder_temp = os.path.join(os.getcwd(), 'video', 'temp')
folder_output = os.path.join(os.getcwd(), 'video', 'output')


# gets the most recent iteration for a certain model build
def most_recent_iteration(build):
    files = os.listdir(os.path.join('builds', build, 'snapshots'))
    mostrecentiteration = 0
    for name in files:
        if name.lower().endswith('.caffemodel'):
            iteration = re.sub('\.caffemodel$', '', name)
            iteration = re.sub('ssd512x512_iter_', '', iteration)
            iteration = re.sub('ssd300x300_iter_', '', iteration)
            if (int(iteration) >= mostrecentiteration):
                mostrecentmodel = name
                mostrecentiteration = int(iteration)
    return mostrecentmodel


# make temp and output folder if it does not exist
if not os.path.exists(folder_temp):
    os.makedirs(folder_temp)
if not os.path.exists(folder_output):
    os.makedirs(folder_output)


# build DeepDetect model repo
if not os.path.exists('dedemodel'):
    os.makedirs('dedemodel')
# copy static files
if not os.path.exists('dedemodel/deploy.prototxt'):
    shutil.copy2('includes/dede_deploy.prototxt', 'dedemodel/deploy.prototxt')
if not os.path.exists('dedemodel/corresp.txt'):
    shutil.copy2('includes/corresp.txt', 'dedemodel/corresp.txt')
# remove old models
for root, dirs, files in os.walk('dedemodel'):
    for name in files:
        if name.lower().endswith('.caffemodel'):
            os.remove(os.path.join(root, name))
# copy new model
recentmodel = most_recent_iteration(args.builddir)
print ('Using model ' + recentmodel)
shutil.copy2(os.path.join('builds', args.builddir, 'snapshots', recentmodel), 'dedemodel/model.caffemodel')


# recursively process directory
for root, dirs, files in os.walk(folder_input):
    for name in files:
        name, ext = os.path.splitext(name)
        if (ext.lower().endswith(('.mp4', '.avi', '.mov')) and os.path.exists(os.path.join(root,name+'.txt'))):


            # video -> jpg_unannotated
            print ('Processing video '+name+'...')
            print ('  Converting video into unannotated frames...')
            # create directories if they do not exist
            folder_video_output = os.path.join(folder_output,name)
            if not os.path.exists(folder_video_output):
                os.makedirs(folder_video_output)
            output_jpg_unannotated = os.path.join(folder_temp,'jpg_unannotated')
            if not os.path.exists(output_jpg_unannotated):
                os.makedirs(output_jpg_unannotated)
            output_jpg_annotated = os.path.join(folder_video_output,'jpg_annotated')
            if not os.path.exists(output_jpg_annotated):
                os.makedirs(output_jpg_annotated)
            output_ssd = os.path.join(folder_video_output,'ssd')
            if not os.path.exists(output_ssd):
                os.makedirs(output_ssd)
            output_csv = os.path.join(folder_video_output,name+'.csv')
            # call ffmpeg
            cmd = 'ffmpeg -nostats -loglevel 0 -i "'+root+'/'+name+ext+'" -r '+str(args.framerate)+' "'+output_jpg_unannotated+'/'+name+'"_%4d.jpg'
            #os.system(cmd)


            # txt -> csv
            print ('  Converting tagging txt into csv...')
            with open(os.path.join(root,name+'.txt')) as txt:
                with open(output_csv, 'w+') as csv:
                    # empty csv file
                    csv.truncate()
                    # original situation
                    minutes = 0
                    seconds = 0
                    status = 2
                    # process per line
                    for line in txt:
                        line = line.replace(' ', '')
                        if (len(line.strip()) != 0):
                            # translate human format to machine format
                            line = line.replace('nodig', '0')
                            line = line.replace('dig', '1')
                            line = line.replace(':', '')
                            # if the assertion fails, the tag file contains an error
                            assert len(line.strip()) == 5
                            # grab from line
                            prevminutes = minutes
                            prevseconds = seconds
                            prevstatus = status
                            minutes = line[0:2]
                            seconds = line[2:4]
                            status = line[4:5]
                            # calculate and push a new csv line, if not the first grab
                            if (prevstatus != 2):
                                pushstatus = prevstatus
                                pushlength = (int(seconds) + int(minutes) * 60) - (int(prevseconds) + int(prevminutes) * 60)
                                csv.write(str(pushlength)+','+str(pushstatus))
                                csv.write('\n')


            # jpg_unannotated -> json, jpg_annotated
            print ('  Processing unannotated frames through DeepDetect...')
            #for subroot, subdirs, subfiles in os.walk(output_jpg_unannotated):
            #    for subname in subfiles: