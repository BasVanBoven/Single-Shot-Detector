#!/usr/bin/python
# seqproc_setup.py - processes input data for training a Sequence Processor

# input: a video folder containing json files: bounding boxes, tags and video resolution
# prerequisites: sudo pip install sklearn


# imports
import json
import os
import pickle
import time
import itertools
import csv
import argparse
import shutil
import random
import numpy as np
import matplotlib.pyplot as plt
import seqproc_common as sp
from random import shuffle
from pprint import pprint
from scipy.stats import mode


# handle input arguments
parser = argparse.ArgumentParser(description='Process input data for a Sequence Processor.')
parser.add_argument('-d', '--debug', default=False, action='store_true', help='print debug output')
parser.add_argument('-a', '--augment', default=False, action='store_true', help='use augmented data for training')
parser.add_argument('-p', '--permutations', type=int, default=10, help='number of augmentation permutations to be generated')
parser.add_argument('-w', '--window', type=int, default=5, help='window size to be used, needs to be an odd number')
parser.add_argument('-c', '--crossval', type=int, default=5, help='number of cross validation splits')
parser.add_argument('-t', '--test', type=float, default=0.2, help='percentage of videos in test set')
parser.add_argument('-s', '--stop', default=False, action='store_true', help='do not start training after setup')
parser.add_argument('-n', '--noserv', default=False, action='store_true', help='do not start serving after training')
args = parser.parse_args()
# window size needs to be uneven to make the majority vote function correctly
assert(args.window % 2 != 0)


# general pathing
rootdir = os.getcwd()
input_boxes = os.path.join(rootdir, 'video', 'output', 'json')
input_tags = os.path.join(rootdir, 'video', 'output', 'tags')
input_resolution = os.path.join(rootdir, 'video', 'output', 'resolution')
output_classifications = os.path.join(rootdir, 'seqproc', '01_classifications')
output_windows = os.path.join(rootdir, 'seqproc', '02_windows')
output_traintest = os.path.join(rootdir, 'seqproc', '03_traintest')
output_train = os.path.join(output_traintest, 'train.csv')
output_test = os.path.join(output_traintest, 'test.csv')
output_train_augmented = os.path.join(output_traintest, 'train_augmented.csv')


# initialize output directories
output_folders = [output_classifications, output_windows, output_traintest]
for folder in output_folders:
    if (os.path.exists(folder)):
        shutil.rmtree(folder)
    os.makedirs(folder, 0755)


# tags -> classification
print 'Converting tags CSV to classification CSV...'
for root, dirs, files in os.walk(input_boxes):
    for video in sorted(dirs):
        # video specific pathing
        tags_csv = os.path.join(input_tags, video+'.csv')
        classifications_csv = os.path.join(output_classifications, video+'.csv')
        # open tags csv file
        tags = np.genfromtxt(tags_csv, delimiter=',', dtype=int)
        # determine the number of frames
        number_of_frames = len(os.listdir(os.path.join(input_boxes, video)))
        # define classification list
        classifications = []
        # for 2D-arrays, i.e., when there is more than one tag for the video
        if tags.size != 2:
            # push classifications
            for tag in range(1, tags.shape[0]):
                for i in range(tags[tag][0] - tags[tag-1][0]):
                    classifications.append(tags[tag-1][1])
            # push the last classification series
            tag = tag + 1
            for i in range((number_of_frames+1) - tags[tag-1][0]):
                classifications.append(tags[tag-1][1])
        # for 1D-arrays, i.e., when there is only one tag for the whole video
        else:
            for i in range(number_of_frames):
                classifications.append(tags[1])
        # write classifications
        with open(classifications_csv, 'wb') as f:
            writer = csv.writer(f)
            writer.writerow(classifications)


# video frame jsons -> frame csv
print 'Converting video frame JSONs to window CSVs...'
for root, dirs, files in os.walk(input_boxes):
    for video in sorted(dirs):
        # video specific pathing
        classifications_csv = os.path.join(output_classifications, video+'.csv')
        resolution_csv = os.path.join(input_resolution, video+'.csv')
        boxes_folder = os.path.join(input_boxes, video)
        # determine video resolution
        resolution = np.genfromtxt(resolution_csv, delimiter=',', dtype=int)
        res_x = resolution[0]
        res_y = resolution[1]
        # open classifications csv file
        classifications = np.genfromtxt(classifications_csv, delimiter=',', dtype=int)
        # initialize window buffer
        json_batch = []
        classificationbuffer = []
        bufferlength = 0
        # open output file, i.e., the window csv
        with open(os.path.join(output_windows, video+'.csv'), 'wb') as f:
            writer = csv.writer(f)
            # do for each frame
            for i, frame in enumerate(sorted(os.listdir(boxes_folder))):
                # open json
                frame_data = json.load(open(os.path.join(boxes_folder, frame), 'r'))
                # if we find a 'unusable' classification, disregard whole window
                if classifications[i] == 2:
                    json_batch = []
                    classificationbuffer = []
                    bufferlength = 0
                    continue
                # otherwise, add the frame to the window
                json_batch.extend([frame_data])
                classificationbuffer.extend([classifications[i]])
                bufferlength += 1
                # if the bufferlength equals window size
                if bufferlength == args.window:
                        if args.augment:
                            # build different permutations
                            for i in range(args.permutations):
                                # write classification and corresponding window to file
                                writer.writerow(np.append(mode(classificationbuffer)[0],sp.window(res_x, res_y, json_batch, True)))
                        else:
                            # write classification and corresponding window to file
                            writer.writerow(np.append(mode(classificationbuffer)[0],sp.window(res_x, res_y, json_batch, False)))
                    # clear the window buffer
                    json_batch = []
                    classificationbuffer = []
                    bufferlength = 0


# window csvs -> train/test split
print 'Converting window CSVs to train/test split...'
# determine the number of test videos
number_train_vids = int(len(os.listdir(output_windows)) * (1-args.test))
vids_list = os.listdir(output_windows)
random.shuffle(vids_list)
train_list = vids_list[0:number_train_vids]
test_list = vids_list[number_train_vids:]
# write train output file
with open(output_train, 'wb') as f:
    for filename in train_list:
        with open(os.path.join(output_windows, filename)) as infile:
            for line in infile:
                f.write(line)
# write test output file
with open(output_test, 'wb') as f:
    for filename in test_list:
        with open(os.path.join(output_windows, filename)) as infile:
            for line in infile:
                f.write(line)
# write test list to csv
with open(os.path.join(output_traintest, 'test_list.csv'), 'wb') as f:
    writer = csv.writer(f)
    writer.writerow(test_list)


# start training
if args.stop == False:
    # build command
    cmd = 'python '+rootdir+'/seqproc_train.py'
    cmd += ' -w '
    cmd += str(args.window)
    cmd += ' -c '
    cmd += str(args.crossval)
    if args.augment == True:
        cmd += ' -a'
    if args.noserv == True:
        cmd += ' -n'
    if args.debug == True:
        cmd += ' -d'
    os.system(cmd)