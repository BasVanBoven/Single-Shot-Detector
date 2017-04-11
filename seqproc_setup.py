#!/usr/bin/python
# setup.py - processes input data for training a Single Shot Detector

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
from random import shuffle
from pprint import pprint
from scipy.stats import mode
from sklearn.model_selection import train_test_split,cross_val_score
from sklearn.metrics import classification_report,confusion_matrix,fbeta_score
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier


# handle input arguments
parser = argparse.ArgumentParser(description='Process input data for training a Sequence Processor.')
parser.add_argument('-a', '--augment', default=True, action='store_true', help='use augmented data for training')
parser.add_argument('-p', '--permutation', type=int, default=10, help='number of augmentation permutations to be generated')
parser.add_argument('-w', '--window', type=int, default=5, help='window size to be used, needs to be an odd number')
parser.add_argument('-c', '--crossval', type=int, default=5, help='number of cross validation splits')
parser.add_argument('-t', '--test', type=int, default=0.1, help='percentage of videos in test set')
args = parser.parse_args()
# window size needs to be uneven to make the majority vote function correctly
assert(args.window % 2 != 0)


# general pathing
rootdir = os.getcwd()
input_boxes = os.path.join(rootdir, 'video', 'output', 'json')
input_tags = os.path.join(rootdir, 'video', 'output', 'tags')
input_resolution = os.path.join(rootdir, 'video', 'output', 'resolution')
output_classifications = os.path.join(rootdir, 'seqproc', '01_classifications')
output_frames = os.path.join(rootdir, 'seqproc', '02_frames')
output_windows = os.path.join(rootdir, 'seqproc', '03_windows')
output_traintest = os.path.join(rootdir, 'seqproc', '04_traintest')
output_train = os.path.join(output_traintest, 'train.csv')
output_test = os.path.join(output_traintest, 'test.csv')
output_train_augmented = os.path.join(output_traintest, 'train_augmented.csv')


# global parameters
number_of_vids = len(os.listdir(input_boxes))


# initialize output directories
output_folders = [output_classifications, output_frames, output_windows, output_traintest]
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


# video frame jsons -> frames csv
print 'Converting video frame JSON to frames CSV...'
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
        # do for each frame
        for i, frame in enumerate(sorted(os.listdir(boxes_folder))):
            # get the strongest detection for each category
            frame_data = json.load(open(os.path.join(boxes_folder, frame), 'r'))
            object_dict = {}
            for detected_object in frame_data['body']['predictions'][0]['classes']:
                category = detected_object['cat']
                if category in object_dict:
                    if object_dict[category]['prob'] < detected_object['prob']:
                        object_dict[category] = detected_object
                else:
                    object_dict[category] = detected_object
            # write classification as first line item
            detections = [classifications[i]]
            # take only excavator parts to the sequence processor
            ordering = ['cabin', 'forearm', 'upperarm', 'wheelbase', 'attachment-bucket', 'attachment-breaker']
            for item in ordering:
                if item in object_dict:
                    # translate to new format
                    obj = object_dict[item]
                    C_X = ((obj['bbox']['xmax'] - obj['bbox']['xmin']) / 2 + obj['bbox']['xmin']) / res_x
                    C_Y = ((obj['bbox']['ymin'] - obj['bbox']['ymax']) / 2 + obj['bbox']['ymax']) / res_y
                    W = (obj['bbox']['xmax'] - obj['bbox']['xmin']) / res_x
                    H = (obj['bbox']['ymin'] - obj['bbox']['ymax']) / res_y
                    Conf = obj['prob']
                    detections.extend([C_X, C_Y, W, H, Conf])
                else:
                    # when an excavator part is not detected
                    detections.extend([0,0,0,0,0])
            # write detections
            with open(os.path.join(output_frames, video+'.csv'), 'a') as f:
                writer = csv.writer(f)
                writer.writerow(detections)


# frames csv -> windows csv
print 'Converting frames CSV to window CSV...'
for filename in sorted(os.listdir(output_frames)):
    # open frames csv file and make sure it contains more than one frame
    frames = np.genfromtxt(os.path.join(output_frames, filename), delimiter=',')
    assert(frames.size > 31)
    # open output file, i.e., the window csv
    with open(os.path.join(output_windows, filename), 'wb') as f:
        writer = csv.writer(f)
        # initialize window buffer
        windowbuffer = []
        classificationbuffer = []
        bufferlength = 0
        # do for each frame, i.e., each line
        for i in range(frames.shape[0]):
            # if we find a 'unusable' classification, disregard whole window
            if frames[i][0] == 2:
                windowbuffer = []
                classificationbuffer = []
                bufferlength = 0
                continue
            # otherwise, add the frame to the window
            bufferlength = bufferlength + 1
            classificationbuffer.extend([frames[i][0]])
            for j in range(1, frames[i][:].shape[0]):
                # difference only from second element, otherwise, fill zeroes
                if bufferlength == 1:
                    windowbuffer.extend(np.append([frames[i][j]], [0]))
                else:
                    windowbuffer.extend(np.append([frames[i][j]], [frames[i][j] - frames[i-1][j]]))
            # if the bufferlength equals window size
            if bufferlength == args.window:
                # write window to file
                #print windowbuffer
                writer.writerow(np.append(mode(classificationbuffer)[0],windowbuffer))
                # clear the window buffer
                windowbuffer = []
                classificationbuffer = []
                bufferlength = 0


# window csv -> train/test split
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


# train/test split -> augmented train/test split
print 'Augmenting train split...'
#TODO