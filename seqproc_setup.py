#!/usr/bin/python
# seqproc_setup.py - processes input data for training a Sequence Processor

# input: a video folder containing json files: bounding boxes, tags and video resolution
# prerequisites: sudo pip install sklearn


# imports
import json
import os
import sys
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
from random import randint
from pprint import pprint
from scipy.stats import mode


# handle input arguments
parser = argparse.ArgumentParser(description='Process input data for a Sequence Processor.')
parser.add_argument('-d', '--debug', default=False, action='store_true', help='print debug output')
parser.add_argument('-a', '--augment', default=False, action='store_true', help='use augmented data for training')
parser.add_argument('-p', '--permutations', type=int, default=10, help='number of augmentation permutations to be generated')
parser.add_argument('-w', '--window', type=int, default=5, help='window size to be used, needs to be an odd number')
parser.add_argument('-e', '--estimator', type=int, default=3, help='estimator that is to be used')
parser.add_argument('-m', '--model', type=str, default='', help='ssd model which determines the blacklist')
parser.add_argument('-l', '--testlist', default=False, action='store_true', help='open testing whitelist for consistent experiments')
parser.add_argument('-c', '--crossval', type=int, default=10, help='number of cross validation splits')
parser.add_argument('-t', '--test', type=float, default=0.2, help='percentage of videos in test set')
parser.add_argument('-s', '--stop', default=False, action='store_true', help='do not start training after setup')
parser.add_argument('-n', '--noserv', default=False, action='store_true', help='do not start serving after training')
args = parser.parse_args()
# window size needs to be uneven to make the majority vote function correctly
assert(args.window % 2 != 0)


# general pathing
rootdir = os.getcwd()
ssd_rootdir = os.path.join(rootdir, 'builds')
input_video = os.path.join(os.getcwd(), 'video', 'input')
input_boxes = os.path.join(rootdir, 'video', 'output', 'json')
input_resolution = os.path.join(rootdir, 'video', 'output', 'resolution')
output_tags = os.path.join(rootdir, 'seqproc', '00_tags')
output_classifications = os.path.join(rootdir, 'seqproc', '01_classifications')
output_windows_clean = os.path.join(rootdir, 'seqproc', '02_windows', 'clean')
output_windows_augmented = os.path.join(rootdir, 'seqproc', '02_windows', 'augmented')
output_traintest = os.path.join(rootdir, 'seqproc', '03_traintest')
output_train = os.path.join(output_traintest, 'train.csv')
output_train_augmented = os.path.join(output_traintest, 'train_augmented.csv')
output_test = os.path.join(output_traintest, 'test.csv')


# determine which ssd build to use for the blacklist
for current in sorted(os.listdir(ssd_rootdir)):
    if current != 'crawl':
        ssd_build = current
if args.model != '':
    assert(os.path.exists(os.path.join(ssd_rootdir, args.model)))
    ssd_build = args.model
# build video test blacklist, cumbersome solution because of backwards compatibility
# it also does not function correctly for videos sourced from YouTube due to weird naming
blacklist = []
for frame in sorted(os.listdir(os.path.join(ssd_rootdir, ssd_build, 'trainval', 'image'))):
    if frame.rsplit('_',1)[0] not in blacklist:
        blacklist.append(frame.rsplit('_',1)[0])

# build test set whitelist if necessary
whitelist_test_set = []
if args.testlist:
    with open('seqproc_test_list.csv', 'r') as whitelist:
        reader = csv.reader(whitelist)
        whitelist_test_set = list(reader)
        whitelist_test_set = whitelist_test_set[0]

# initialize output directories
output_folders = [output_classifications, output_windows_clean, output_windows_augmented, output_traintest, output_tags]
for folder in output_folders:
    if (os.path.exists(folder)):
        shutil.rmtree(folder)
    os.makedirs(folder, 0755)


# assert the txt files directly, just to be sure they are readable
print ('Converting tags txt to tags csv')
for root, dirs, files in os.walk(input_video):
    for name in files:
        name, ext = os.path.splitext(name)
        if ext.lower().endswith('.txt'):
            #print ('Processing tags for '+name+'...')
            with open(os.path.join(root,name+'.txt')) as txt:
                with open(os.path.join(output_tags,name+'.csv'), 'w+') as tagscsv:
                    # empty csv file
                    tagscsv.truncate()
                    # process per line
                    for line in txt:
                        line = line.replace(' ', '')
                        if (len(line.strip()) != 0):
                            # translate human format to machine format
                            # modify these lines to run various experiments
                            line = line.replace('nodig', '0')
                            line = line.replace('drive', '0')
                            line = line.replace('rotate', '0')
                            line = line.replace('lowerraise', '0')
                            line = line.replace('standstill', '0')
                            line = line.replace('notinscene', '0')
                            line = line.replace('dig', '1')
                            line = line.replace('unusable', '2')
                            line = line.replace(':', '')
                            # if the assertion fails, the tag file contains an error
                            assert len(line.strip()) == 5
                            # grab from line
                            minutes = line[0:2]
                            seconds = line[2:4]
                            status = line[4:5]
                            # calculate and push a new csv line
                            pushlength = (int(seconds) + int(minutes) * 60) + 1
                            tagscsv.write(str(pushlength)+','+str(status))
                            tagscsv.write('\n')


# tags -> classification
print 'Converting tags CSV to classification CSV...'
for root, dirs, files in os.walk(output_tags):
    for video in sorted(files[:-4]):
        # strip extension
        video = video[:-4]
        # video specific pathing
        tags_csv = os.path.join(output_tags, video+'.csv')
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
for root, dirs, files in os.walk(output_tags):
    for video in sorted(files[:-4]):
        # strip extension
        video = video[:-4]
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
        with open(os.path.join(output_windows_clean, video+'.csv'), 'wb') as c, open(os.path.join(output_windows_augmented, video+'.csv'), 'wb') as a:
            # define output writers
            clean = csv.writer(c)
            augmented = csv.writer(a)
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
                    # write classification and corresponding window to file
                    clean.writerow(np.append(mode(classificationbuffer)[0],sp.window(res_x, res_y, json_batch, False)))
                    # also build permutations if desired
                    if args.augment:
                        # build different permutations
                        for i in range(args.permutations):
                            # write classification and corresponding window to file
                            augmented.writerow(np.append(mode(classificationbuffer)[0],sp.window(res_x, res_y, json_batch, True)))
                    # clear the window buffer
                    json_batch = []
                    classificationbuffer = []
                    bufferlength = 0


# window csvs -> train/test split
print 'Converting window CSVs to train/test split...'
# determine the number of test videos and fill lists
number_test_vids = int(len(os.listdir(output_windows_clean)) * (args.test))
vids_list = os.listdir(output_windows_clean)
random.shuffle(vids_list)
test_list = []
train_list = []
# split train/test lists
# two modes: either build split manually or import a test list
# the imported list should have already taken the blacklist into account
if args.testlist:
    test_list = whitelist_test_set
    for vid in vids_list:
        if vid not in test_list:
            train_list.append(vid)
    # assert that the imported list takes the blacklist into account
    for vid in test_list:
        assert (vid not in blacklist)
else:
    while (number_test_vids > len(test_list)):
        index = randint(0,len(vids_list)-1)
        if vids_list[index] not in blacklist:
            test_list.append(vids_list[index])
            vids_list.remove(vids_list[index])
    train_list = vids_list
# write train output file
with open(output_train, 'wb') as f:
    for filename in train_list:
        with open(os.path.join(output_windows_clean, filename)) as infile:
            for line in infile:
                f.write(line)
# write test output file
with open(output_test, 'wb') as f:
    for filename in test_list:
        with open(os.path.join(output_windows_clean, filename)) as infile:
            for line in infile:
                f.write(line)
# write train_augmented output file, if necessary
if args.augment:
    with open(output_train_augmented, 'wb') as f:
        for filename in train_list:
            with open(os.path.join(output_windows_augmented, filename)) as infile:
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
    cmd += ' -e '
    cmd += str(args.estimator)
    cmd += ' -c '
    cmd += str(args.crossval)
    if args.augment == True:
        cmd += ' -a'
    if args.noserv == True:
        cmd += ' -n'
    if args.debug == True:
        cmd += ' -d'
    os.system(cmd)