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
import numpy as np
import pandas as pd
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
parser.add_argument('-w', '--window', type=int, default=5, help='window size to be used')
parser.add_argument('-c', '--crossval', type=int, default=5, help='number of cross validation splits')
parser.add_argument('-t', '--test', type=int, default=0.1, help='percentage of images in test set')
args = parser.parse_args()


# general pathing
rootdir = os.getcwd()
input_boxes = os.path.join(rootdir, 'video', 'output', 'json')
input_tags = os.path.join(rootdir, 'video', 'output', 'tags')
input_resolution = os.path.join(rootdir, 'video', 'output', 'resolution')
output_frames = os.path.join(rootdir, 'seqproc', '01_frames')
output_windows = os.path.join(rootdir, 'seqproc', '02_windows')
output_split_train = os.path.join(rootdir, 'seqproc', '03_split', 'train')
output_split_test = os.path.join(rootdir, 'seqproc', '03_split', 'test')
output_augment_train = os.path.join(rootdir, 'seqproc', '04_augmented', 'train')
output_augment_test = os.path.join(rootdir, 'seqproc', '04_augmented', 'test')


# global parameters
number_of_vids = len(os.listdir(input_boxes))


# initialize output directories
output_folders = [output_frames, output_windows, output_split_train, output_split_test, output_augment_train, output_augment_test]
for folder in output_folders:
    if (os.path.exists(folder)):
        shutil.rmtree(folder)
    os.makedirs(folder, 0755)


# video frame json -> master data csv
print 'Converting video frame JSON to master data CSV...'
for root, dirs, files in os.walk(input_boxes):
    for video in dirs:
        # video specific pathing
        tags_csv = os.path.join(input_tags, video+'.csv')
        resolution_csv = os.path.join(input_resolution, video+'.csv')
        boxes_folder = os.path.join(input_boxes, video)
        # determine video resolution
        res_df = pd.read_csv(resolution_csv, header=None)
        xres = int(res_df[0][0])
        yres = int(res_df[1][0])
        # open tags csv (0,1,2 for nodig, dig and unusable)
        tags_df = pd.read_csv(tags_csv, header=None)
        # remove 1 from first column, for code count (0,1... vs 1,2... )
        tags_df = pd.concat([tags_df[0] - 1,tags_df[1]], axis=1)
        nshape = tags_df.shape[0]
        linecount = 0
        # use tags to determine how to label each json
        current_tag = tags_df.loc[0][1]
        change_index = float('Inf')
        if(linecount+1 < nshape):
            change_index = tags_df.loc[1][0]
        # for each json file in video directory
        for index,filename in enumerate(sorted(os.listdir(boxes_folder))):
            with open(os.path.join(boxes_folder, filename), 'r') as json_in:
                #print filename
                # upon reaching a json where tagging changes
                if index == change_index:
                    linecount = linecount+1
                    current_tag = tags_df.loc[linecount][1]
                    if linecount+1 < nshape:
                        change_index = tags_df.loc[linecount+1][0]
                # get strongest detection for each category
                data = json.load(json_in)
                object_dict = {}
                for detected_object in data['body']['predictions'][0]['classes']:
                    category = detected_object['cat']
                    if category in object_dict:
                        if object_dict[category]['prob'] < detected_object['prob']:
                            object_dict[category] = detected_object
                    else:
                        object_dict[category] = detected_object
                data_array = []
                ordering = ['cabin', 'forearm', 'upperarm', 'wheelbase', 'attachment-bucket', 'attachment-breaker']
                for index,item in enumerate(ordering):
                    if item in object_dict:
                        obj = object_dict[item]
                        C_X = ((obj['bbox']['xmax'] - obj['bbox']['xmin']) / 2 + obj['bbox']['xmin']) / xres
                        C_Y = ((obj['bbox']['ymin'] - obj['bbox']['ymax']) / 2 + obj['bbox']['ymax']) / yres
                        W = (obj['bbox']['xmax'] - obj['bbox']['xmin']) / xres
                        H = (obj['bbox']['ymin'] - obj['bbox']['ymax']) / yres
                        Conf = obj['prob']
                        data_array.extend([C_X, C_Y, W, H, Conf])
                    else:
                        data_array.extend([0,0,0,0,0])
                data_array.extend([current_tag])
                with open(os.path.join(output_frames, video + '.csv'), 'a') as csvfile:
                    writer = csv.writer(csvfile, delimiter=',', lineterminator='\n')
                    rounded = [float(i.round(decimals=5)) for i in np.array(data_array)]
                    # do not write unusable tags
                    if(current_tag != 2):
                        writer.writerow(rounded)
                object_dict.clear()


# master data csv -> window csv
print 'Converting master data CSV to window CSV...'
#TODO


# window csv -> train/test split
print 'Converting window CSV to train/test split...'
#TODO


# train/test split -> augmented train/test split
print 'Augmenting train/test split...'
#TODO